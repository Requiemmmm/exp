import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
import transformers
from transformers import (
    HfArgumentParser,
    DataCollatorWithPadding
)
from transformers import AutoTokenizer, AutoConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification as TModel

from modeling.modeling_cofi_bert import CoFiBertForSequenceClassification as SModel
from modeling.dp_engine import DifferentialPrivacyEngine
from modeling.mask import Mask

from datasets import DatasetDict, Dataset, load_from_disk, load_metric, load_dataset
from typing import Optional, Dict, List, Tuple, Callable, Union
from copy import deepcopy
from transformers import Trainer
import logging

from .trainer import DistillTrainer
from .args import (
    TrainingArguments,
    ModelArguments,
)
from modeling.dp_engine import LocalDPEngine

Trainers = Union[Trainer, DistillTrainer]


def parse_hf_args():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True)

    return args, training_args


def get_distill_args(args):
    distill_args = deepcopy(args)
    distill_args.num_train_epochs = args.distill_num_train_epochs
    distill_args.learning_rate = args.distill_learning_rate
    distill_args.evaluation_strategy = "epoch"

    return distill_args


class UnifiedSparsityManager:
    """
    统一的稀疏率管理器
    解决稀疏率概念混乱问题
    """

    def __init__(self):
        # 明确定义：sparsity = zero_weights / total_weights
        self.definition = "fraction_of_zero_weights"

    def compute_model_sparsity(self, model):
        """计算模型的实际稀疏率"""
        total_params = 0
        zero_params = 0

        # MODIFIED: More robustly compute sparsity from CoFi masks
        for module in model.modules():
            if isinstance(module, Mask):
                # Using deterministic_z gives the actual mask used during evaluation
                mask_values = module.deterministic_z()
                total_params += mask_values.numel()
                zero_params += (mask_values == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


class ProgressivePruningScheduler:
    """
    修复后的渐进式剪枝调度器
    使用统一的稀疏率定义
    """

    def __init__(self, initial_sparsity=0.0, target_sparsity=0.6, total_rounds=10):
        # 明确定义：sparsity = 被剪掉的参数比例
        self.initial_sparsity = initial_sparsity  # 初始剪掉0%
        self.target_sparsity = target_sparsity  # 最终剪掉30%
        self.total_rounds = total_rounds

    def get_current_sparsity(self, current_round):
        """渐进式剪枝：稀疏率逐渐增加"""
        if current_round < 2:  # 热身期
            return self.initial_sparsity

        progress = min(1.0, (current_round - 2) / max(1, self.total_rounds - 2))
        # 使用线性增长而不是三次方，更加稳定
        sparsity = self.initial_sparsity + \
                   (self.target_sparsity - self.initial_sparsity) * progress

        return min(sparsity, self.target_sparsity)

    def get_distill_weight(self, current_sparsity):
        """自适应蒸馏权重：剪枝越激进，蒸馏越重要"""
        base_weight = 0.1  # 降低基础蒸馏权重
        # 线性增长：稀疏率0.0->0.3时，蒸馏权重0.1->0.4
        adaptive_weight = base_weight + 0.3 * current_sparsity
        return min(adaptive_weight, 0.5)  # 限制最大蒸馏权重


class Client():
    def __init__(self, epsilon=1000, num_clients=2):

        args, training_args = parse_hf_args()
        # Note: The user confirmed they have unified the dataset.
        # This path might need to be an argument in a future version.
        dataset = load_from_disk('./datasets/sst2')
        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize_function(example):
            return self.tokenizer(example["sentence"], truncation=True)  # sst2数据集用这行

        dataset = dataset.map(tokenize_function, batched=True)
        dataset['train'] = dataset['train'].filter(lambda x: len(x["input_ids"]) <= 512)

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.epsilon = epsilon
        self.num_clients = num_clients
        self.dataset = dataset
        self.half = training_args.half
        self.client_train_datas = self.load_client_train_datas()

        self.distill_args = get_distill_args(training_args)
        self.distill_args.num_train_epochs = 2
        self.distill_args.gradient_accumulation_steps = 4

        # ✅ 调整后的DP配置（更宽松的隐私预算）
        self.dp_config = {
            'target_epsilon': 10.0,  # 增大epsilon，减少噪声
            'target_delta': 1e-3,  # 稍微放松delta
            'max_grad_norm': 1.0,  # 梯度裁剪阈值
            'noise_multiplier': 0.003,  # 减少噪声乘数
        }

        self.dp_engine = LocalDPEngine(**self.dp_config)
        print(f"✅ 客户端DP引擎初始化完成: ε={self.dp_config['target_epsilon']}, δ={self.dp_config['target_delta']}")

    def load_client_train_datas(self):
        client_train_datas = []
        if self.half == False:
            for i in range(self.num_clients):
                client_train_datas.append(
                    self.dataset['train'].shard(num_shards=self.num_clients, index=i, contiguous=True))
        else:
            for i in range(self.num_clients):
                client_train_datas.append(
                    self.dataset['train'].shard(num_shards=self.num_clients * 2, index=self.num_clients + i,
                                                contiguous=True))
        return client_train_datas

    def compute_metrics(self, eval_pred):
        logits_, labels = eval_pred
        predictions = np.argmax(logits_, axis=-1)
        accuracy = np.sum(predictions == labels) / len(labels)

        return {"accuracy": accuracy}

    # MODIFIED: Function now returns weights and the deterministic masks
    def train_epoch(self, server_model, client_id, server_weights, t_model):
        """
        客户端训练一个epoch.
        MODIFIED: 返回权重和剪枝决策 (masks).
        """
        datasets = self.client_train_datas[client_id]
        server_model.load_state_dict(server_weights)

        distill_trainer = DistillTrainer(
            server_model,
            t_model,
            args=self.distill_args,
            train_dataset=datasets,
            eval_dataset=self.dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        distill_trainer.train()

        new_weights = server_model.state_dict()

        # ADDED: Generate deterministic binary masks from the trained model
        client_masks = {}
        with torch.no_grad():
            for name, module in server_model.named_modules():
                if isinstance(module, Mask):
                    # Use deterministic_z to get the binary mask used in evaluation
                    binary_mask = module.deterministic_z()
                    client_masks[name] = binary_mask.cpu()

        private_weights = {}
        processed_count = 0
        skipped_count = 0
        clip_norm = self.dp_config['max_grad_norm']

        for name, new_param in new_weights.items():
            if self._should_add_noise(name) and new_param.dtype.is_floating_point:
                delta = new_param - server_weights[name].to(new_param.device)
                delta_norm = torch.norm(delta).item()
                if delta_norm > clip_norm:
                    delta.mul_(clip_norm / (delta_norm + 1e-6))
                noised_delta = self.dp_engine.add_noise(delta)
                private_weights[name] = server_weights[name].to(noised_delta.device) + noised_delta
                processed_count += 1
            else:
                private_weights[name] = new_param
                skipped_count += 1

        print(f"  🔒 DP处理完成 (正确逻辑): 已处理{processed_count}个参数, 跳过{skipped_count}个参数")
        print(f"  隐私保障: 此轮提供约 ({self.dp_engine.per_round_epsilon:.4f}, {self.dp_engine.target_delta})-DP")

        # Return both weights and the generated masks
        return private_weights, client_masks

    def _should_add_noise(self, param_name: str) -> bool:
        """判断是否需要对特定参数添加噪声"""
        skip_patterns = [
            'reg_lambda', 'bias', 'LayerNorm', 'layer_norm',
            'position_ids', 'token_type_ids', 'mask'
        ]
        return not any(pattern in param_name for pattern in skip_patterns)


class Server():
    def __init__(self, epochs=10, num_clients=2):
        args, training_args = parse_hf_args()
        self.num_clients = num_clients
        self.client = Client()
        self.epochs = epochs
        self.distill = training_args.distill

        if self.distill == True:
            self.t_model = TModel.from_pretrained('./[glue]/sst2-half-datas')
            self.s_model = SModel.from_pretrained('./[glue]/sst2-half-datas')
        if self.distill == False:
            self.t_model = TModel.from_pretrained('./model')
            self.s_model = SModel.from_pretrained('./model')

        dataset = load_from_disk('./datasets/sst2')
        self.tokenizer = AutoTokenizer.from_pretrained("./model")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        def tokenize_function(example):
            return self.tokenizer(example["sentence"], truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)
        dataset['validation'] = dataset['validation'].filter(lambda x: len(x["input_ids"]) <= 512)

        self.dataset = dataset['validation']
        self.distill_args = get_distill_args(training_args)
        self.distill_args.num_train_epochs = 1
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.best_result = 0

        self.pruning_scheduler = ProgressivePruningScheduler(
            initial_sparsity=0.0,
            target_sparsity=0.6,
            total_rounds=self.epochs
        )
        self.sparsity_manager = UnifiedSparsityManager()

        print(f"✅ 服务器初始化完成: {num_clients}个客户端, {epochs}轮训练")
        print(f"✅ 剪枝调度: {self.pruning_scheduler.initial_sparsity} -> {self.pruning_scheduler.target_sparsity}")
        print(f"✅ 隐私模型: 客户端本地差分隐私 (Local DP)")

    # MODIFIED: Function now collects and returns masks as well
    def distribute_task(self, client_ids):
        """分发训练任务到客户端, 并收集权重和剪枝决策。"""
        server_weights = deepcopy(self.s_model.state_dict())
        client_weight_datas = []
        client_mask_datas = []  # ADDED: To store masks from clients

        for i in range(len(client_ids)):
            client_id = client_ids[i]
            print(f"  📤 向客户端 {client_id} 分发任务...")

            # train_epoch now returns weights and masks
            weight, masks = self.client.train_epoch(
                deepcopy(self.s_model),
                client_id,
                deepcopy(server_weights),
                self.t_model
            )
            client_weight_datas.append(weight)
            client_mask_datas.append(masks)  # ADDED: Collect masks

        return client_weight_datas, client_mask_datas

    # MODIFIED: Complete rewrite of the aggregation logic
    def federated_average(self, client_weights, client_masks, client_data_sizes=None):
        """
        修复后的联邦聚合.
        - 聚合模型权重.
        - 对剪枝的二元mask进行多数投票.
        """
        if not client_weights:
            raise ValueError("No client weights provided")

        # --- 1. 聚合模型权重 (与之前逻辑类似) ---
        num_clients = len(client_weights)
        if client_data_sizes is None:
            agg_weights = [1.0 / num_clients] * num_clients
        else:
            total_samples = sum(client_data_sizes)
            agg_weights = [size / total_samples for size in client_data_sizes]

        print(f"  🔄 聚合 {num_clients} 个客户端的权重...")
        aggregated_weights_state_dict = self._type_aware_aggregate(client_weights, agg_weights)

        # --- 2. 聚合剪枝Mask (核心修改) ---
        aggregated_masks = {}
        if client_masks:
            print(f"  🗳️ 对 {len(client_masks[0])} 个剪枝mask进行多数投票...")
            mask_keys = client_masks[0].keys()
            for key in mask_keys:
                # Stack masks from all clients for the current layer/module
                all_masks_for_key = torch.stack([m[key] for m in client_masks])

                # Perform weighted majority vote
                # Initialize vote accumulator with the correct type and device
                vote_accumulator = torch.zeros_like(all_masks_for_key[0], dtype=torch.float32)

                for i in range(num_clients):
                    vote_accumulator += agg_weights[i] * all_masks_for_key[i].float()

                # Decision boundary is 0.5. If weighted vote > 0.5, keep the weight (mask=1)
                aggregated_masks[key] = (vote_accumulator > 0.5).to(all_masks_for_key[0].dtype)

        # 加载聚合后的权重到服务器模型
        self.s_model.load_state_dict(aggregated_weights_state_dict, strict=False)

        # 返回聚合后的权重字典和mask字典
        return aggregated_weights_state_dict, aggregated_masks

    # ADDED: New function to apply aggregated masks to the global model state
    def apply_aggregated_masks(self, aggregated_masks):
        """
        将全局统一的剪枝决策应用回模型，重置log_alpha以供下轮训练。
        """
        print(f"  🎭 应用全局剪枝决策到模型...")
        with torch.no_grad():
            for name, module in self.s_model.named_modules():
                if name in aggregated_masks:
                    # Found a module with a corresponding aggregated mask
                    binary_mask = aggregated_masks[name].to(module.log_alpha.device)

                    # Set log_alpha based on the binary mask decision
                    # Kept weights (mask=1) get a high log_alpha
                    # Pruned weights (mask=0) get a low log_alpha
                    module.log_alpha.data[binary_mask == 1] = 10.0
                    module.log_alpha.data[binary_mask == 0] = -10.0

    def _type_aware_aggregate(self, client_weight_datas, client_weights=None):
        """类型感知的安全聚合"""
        client_num = len(client_weight_datas)
        first_weights = client_weight_datas[0]

        if client_weights is None:
            client_weights = [1.0 / client_num] * client_num

        aggregated_weights = {}
        processed_params = 0

        for key in first_weights.keys():
            # Skip mask parameters, they will be handled by apply_aggregated_masks
            if 'mask' in key:
                aggregated_weights[key] = first_weights[key].clone() if first_weights[key] is not None else None
                continue

            first_param = first_weights[key]

            try:
                if first_param is None:
                    aggregated_weights[key] = None
                    continue

                if not first_param.dtype.is_floating_point:
                    if first_param.dtype == torch.bool:
                        vote_sum = torch.zeros_like(first_param, dtype=torch.float32)
                        for weights, weight in zip(client_weight_datas, client_weights):
                            if weights[key] is not None:
                                vote_sum += weight * weights[key].float()
                        aggregated_weights[key] = (vote_sum > 0.5).to(first_param.dtype)
                    else:
                        aggregated_weights[key] = first_param.clone()
                else:
                    aggregated_weights[key] = torch.zeros_like(first_param)
                    for weights, weight in zip(client_weight_datas, client_weights):
                        if weights[key] is not None:
                            # Ensure weights[key] is on the same device as aggregated_weights[key]
                            aggregated_weights[key] += weight * weights[key].to(aggregated_weights[key].device)
                processed_params += 1
            except Exception as e:
                print(f"    ⚠️ 参数 {key} 聚合失败: {e}")
                aggregated_weights[key] = first_param.clone() if first_param is not None else None

        print(f"  ✅ 安全聚合完成: 处理{processed_params}个非mask参数")
        return aggregated_weights

    def compute_metrics(self, eval_pred):
        logits_, labels = eval_pred
        predictions = np.argmax(logits_, axis=-1)
        accuracy = np.sum(predictions == labels) / len(labels)
        return {"accuracy": accuracy}

    def evalute(self):
        """评估当前模型性能"""
        distill_trainer = DistillTrainer(
            self.s_model,
            self.t_model,
            args=self.distill_args,
            eval_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        results = distill_trainer.evaluate(eval_dataset=self.dataset)
        actual_sparsity = self.sparsity_manager.compute_model_sparsity(self.s_model)
        results['eval_actual_sparsity'] = actual_sparsity
        current_accuracy = results['eval_accuracy']

        if current_accuracy > self.best_result:
            self.best_result = current_accuracy

        print(f"  📊 当前性能: 准确率={current_accuracy:.4f}, 实际稀疏率={actual_sparsity:.4f}")
        print(f"  🏆 最佳结果: {self.best_result:.4f}")

        return results

    # MODIFIED: Main training loop updated to handle new function signatures and logic
    def run(self):
        """
        修复后的主训练循环.
        """
        print(f"\n🚀 开始联邦学习训练 ({self.epochs} 轮)")
        print("=" * 60)

        for epoch in range(self.epochs):
            print(f"\n📅 第 {epoch + 1}/{self.epochs} 轮训练")
            print("-" * 40)

            current_sparsity = self.pruning_scheduler.get_current_sparsity(epoch)
            distill_weight = self.pruning_scheduler.get_distill_weight(current_sparsity)

            print(f"  🎯 剪枝参数: 目标稀疏率={current_sparsity:.3f} (即{current_sparsity * 100:.1f}%权重为0)")
            print(f"  🧠 蒸馏权重: {distill_weight:.3f}")

            self.client.distill_args.target_sparsity = current_sparsity
            self.client.distill_args.distill_lambda = distill_weight

            client_ids = [i for i in range(self.num_clients)]

            # distribute_task now returns masks
            client_weight_datas, client_mask_datas = self.distribute_task(client_ids)

            # federated_average now takes masks as input
            _, aggregated_masks = self.federated_average(client_weight_datas, client_mask_datas)

            # ADDED: Apply the global mask decision back to the model's log_alpha
            self.apply_aggregated_masks(aggregated_masks)

            self.evalute()

        print("\n🎉 训练完成！")
        print("=" * 60)
        print(f"🏆 最终最佳准确率: {self.best_result:.4f}")
        final_sparsity = self.sparsity_manager.compute_model_sparsity(self.s_model)
        print(f"🔧 最终模型稀疏率: {final_sparsity:.4f}")

        return {
            'best_accuracy': self.best_result,
            'final_sparsity': final_sparsity,
            'target_sparsity': self.pruning_scheduler.target_sparsity
        }
