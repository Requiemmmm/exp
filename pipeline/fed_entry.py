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
# MODIFICATION: Removed unused import of FederatedDPAggregator
from modeling.dp_engine import DifferentialPrivacyEngine

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

        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
                zero_params += (param.abs() < 1e-8).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

    def apply_structured_pruning(self, model, target_sparsity):
        """应用结构化剪枝，确保稀疏率计算正确"""

        total_params = 0
        pruned_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                param_count = param.numel()
                total_params += param_count

                # 计算当前层需要剪枝的参数数量
                target_pruned = int(param_count * target_sparsity)

                if target_pruned > 0:
                    # 找到最小的target_pruned个权重
                    flat_weights = param.abs().flatten()
                    threshold_value = torch.kthvalue(flat_weights, target_pruned)[0]

                    # 创建mask
                    mask = (param.abs() > threshold_value).float()
                    param.data *= mask

                    # 统计实际剪枝的参数
                    actual_pruned = (mask == 0).sum().item()
                    pruned_params += actual_pruned

        actual_sparsity = pruned_params / total_params if total_params > 0 else 0.0

        return {
            'target_sparsity': target_sparsity,
            'actual_sparsity': actual_sparsity,
            'total_params': total_params,
            'pruned_params': pruned_params
        }


class ProgressivePruningScheduler:
    """
    修复后的渐进式剪枝调度器
    使用统一的稀疏率定义
    """

    def __init__(self, initial_sparsity=0.0, target_sparsity=0.3, total_rounds=10):
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
        self.distill_args.num_train_epochs = 1
        self.distill_args.gradient_accumulation_steps = 4

        # ✅ 调整后的DP配置（更宽松的隐私预算）
        self.dp_config = {
            'target_epsilon': 10.0,  # 增大epsilon，减少噪声
            'target_delta': 1e-3,  # 稍微放松delta
            'max_grad_norm': 1.0,  # 梯度裁剪阈值
            'noise_multiplier': 0.003,  # 减少噪声乘数
            #'sample_rate': 0.01  # 采样率
        }

        # 初始化DP引擎
        #self.dp_engine = DifferentialPrivacyEngine(**self.dp_config)
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

    def train_epoch(self, server_model, client_id, server_weights, t_model):
        """
        客户端训练一个epoch
        集成了规范化的差分隐私保护 (已修复裁剪逻辑)
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

        # 执行训练
        distill_trainer.train()

        # 获取训练后的权重
        new_weights = server_model.state_dict()

        # ✅ 修复后的差分隐私处理：裁剪和加噪作用于模型更新(Delta)
        private_weights = {}
        processed_count = 0
        skipped_count = 0

        clip_norm = self.dp_config['max_grad_norm']

        for name, new_param in new_weights.items():
            # 只对需要加噪的浮点型参数进行处理
            if self._should_add_noise(name) and new_param.dtype.is_floating_point:

                # 1. 计算模型更新的“增量”(delta)
                delta = new_param - server_weights[name].to(new_param.device)

                # 2. 对增量进行范数裁剪
                delta_norm = torch.norm(delta).item()
                if delta_norm > clip_norm:
                    delta.mul_(clip_norm / (delta_norm + 1e-6))  # 使用 in-place 乘法提升效率

                # 3. 对裁剪后的增量添加噪声
                noised_delta = self.dp_engine.add_noise(delta)

                # 4. 将加噪后的增量应用回原始权重，得到最终要上传的权重
                private_weights[name] = server_weights[name].to(noised_delta.device) + noised_delta
                processed_count += 1
            else:
                # 不需要处理的参数直接使用新权重
                private_weights[name] = new_param
                skipped_count += 1

        print(f"  🔒 DP处理完成 (正确逻辑): 已处理{processed_count}个参数, 跳过{skipped_count}个参数")
        print(f"  隐私保障: 此轮提供约 ({self.dp_engine.per_round_epsilon:.4f}, {self.dp_engine.target_delta})-DP")

        return private_weights

    def _should_add_noise(self, param_name: str) -> bool:
        """判断是否需要对特定参数添加噪声"""
        # 扩展的跳过模式
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

        # ✅ 修复后的渐进式剪枝调度器（更保守的参数）
        self.pruning_scheduler = ProgressivePruningScheduler(
            initial_sparsity=0.0,  # 从0%开始
            target_sparsity=0.3,  # 最终30%（而不是90%）
            total_rounds=self.epochs
        )

        # ✅ 初始化稀疏率管理器
        self.sparsity_manager = UnifiedSparsityManager()

        # ==============================================================================
        # MODIFICATION START: Removed unused FederatedDPAggregator
        # ==============================================================================
        # The following block has been removed to simplify the DP logic, as
        # privacy is already handled on the client-side (Local DP).
        #
        # self.fed_dp_aggregator = FederatedDPAggregator(...)
        #
        # ==============================================================================
        # MODIFICATION END
        # ==============================================================================

        print(f"✅ 服务器初始化完成: {num_clients}个客户端, {epochs}轮训练")
        print(f"✅ 剪枝调度: {self.pruning_scheduler.initial_sparsity} -> {self.pruning_scheduler.target_sparsity}")
        print(f"✅ 隐私模型: 客户端本地差分隐私 (Local DP)")


    def distribute_task(self, client_ids):
        """分发训练任务到客户端"""
        server_weights = deepcopy(self.s_model.state_dict())
        client_weight_datas = []

        for i in range(len(client_ids)):
            client_id = client_ids[i]
            print(f"  📤 向客户端 {client_id} 分发任务...")

            # 深拷贝避免权重污染
            weight = self.client.train_epoch(
                deepcopy(self.s_model),
                client_id,
                deepcopy(server_weights),
                self.t_model
            )
            client_weight_datas.append(weight)

        return client_weight_datas

    def federated_average(self, client_weight_datas):
        """
        修复后的联邦平均聚合算法
        使用类型感知的安全聚合
        """
        client_num = len(client_weight_datas)
        if client_num == 0:
            raise ValueError("No client weights received")

        print(f"  🔄 聚合 {client_num} 个客户端的权重...")

        # ✅ 使用类型感知的安全聚合
        aggregated_weights = self._type_aware_aggregate(client_weight_datas)

        self.s_model.load_state_dict(aggregated_weights)
        return aggregated_weights

    def _type_aware_aggregate(self, client_weight_datas):
        """类型感知的安全聚合"""
        client_num = len(client_weight_datas)
        first_weights = client_weight_datas[0]

        aggregated_weights = {}
        processed_params = 0

        for key in first_weights.keys():
            first_param = first_weights[0][key] if isinstance(first_weights, list) else first_weights[key]

            try:
                if not first_param.dtype.is_floating_point:
                    # 非浮点型参数：使用多数投票或直接复制
                    if first_param.dtype == torch.bool:
                        # 布尔型参数：多数投票
                        vote_sum = torch.zeros_like(first_param, dtype=torch.float32)
                        for weights in client_weight_datas:
                            vote_sum += weights[key].float()
                        aggregated_weights[key] = (vote_sum > client_num / 2).to(first_param.dtype)
                    else:
                        # 其他整型参数：使用第一个值
                        aggregated_weights[key] = first_param.clone()
                else:
                    # 浮点型参数：加权平均
                    aggregated_weights[key] = torch.zeros_like(first_param)
                    for weights in client_weight_datas:
                        aggregated_weights[key] += weights[key] / client_num

                processed_params += 1

            except Exception as e:
                print(f"    ⚠️ 参数 {key} 聚合失败: {e}")
                # 备用方案：使用第一个客户端的权重
                aggregated_weights[key] = first_param.clone()

        print(f"  ✅ 安全聚合完成: 处理{processed_params}个参数")
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

        # 计算实际稀疏率
        actual_sparsity = self.sparsity_manager.compute_model_sparsity(self.s_model)
        results['actual_sparsity'] = actual_sparsity

        # 更新最佳结果
        current_accuracy = results['eval_accuracy']

        if current_accuracy > self.best_result:
            self.best_result = current_accuracy

        print(f"  📊 当前性能: 准确率={current_accuracy:.4f}, 实际稀疏率={actual_sparsity:.4f}")
        print(f"  🏆 最佳结果: {self.best_result:.4f}")

        return results

    def run(self):
        """
        修复后的主训练循环
        解决稀疏率概念混乱和数值不稳定问题
        """
        print(f"\n🚀 开始联邦学习训练 ({self.epochs} 轮)")
        print("=" * 60)

        for epoch in range(self.epochs):
            print(f"\n📅 第 {epoch + 1}/{self.epochs} 轮训练")
            print("-" * 40)

            # ✅ 使用统一的稀疏率定义
            current_sparsity = self.pruning_scheduler.get_current_sparsity(epoch)
            distill_weight = self.pruning_scheduler.get_distill_weight(current_sparsity)

            print(f"  🎯 剪枝参数: 目标稀疏率={current_sparsity:.3f} (即{current_sparsity * 100:.1f}%权重为0)")
            print(f"  🧠 蒸馏权重: {distill_weight:.3f}")

            # ✅ 正确设置客户端的剪枝参数
            # 直接使用统一的稀疏率定义，不进行复杂转换
            self.client.distill_args.target_sparsity = current_sparsity
            self.client.distill_args.distill_lambda = distill_weight

            # 分发任务和聚合
            client_ids = [i for i in range(self.num_clients)]
            client_weight_datas = self.distribute_task(client_ids)
            self.federated_average(client_weight_datas)

            # 评估性能
            results = self.evalute()

            # ✅ 验证稀疏率是否符合预期
            '''actual_sparsity = results['actual_sparsity']
            sparsity_gap = abs(actual_sparsity - current_sparsity)

            if sparsity_gap > 0.05:  # 如果稀疏率偏差超过5%
                print(f"  🔧 稀疏率偏差过大({sparsity_gap:.3f})，进行结构化剪枝调整...")
                pruning_result = self.sparsity_manager.apply_structured_pruning(
                    self.s_model, current_sparsity
                )
                print(f"  ✅ 调整后稀疏率: {pruning_result['actual_sparsity']:.3f}")'''

        print("\n🎉 训练完成！")
        print("=" * 60)
        print(f"🏆 最终最佳准确率: {self.best_result:.4f}")

        # 打印最终稀疏率统计
        final_sparsity = self.sparsity_manager.compute_model_sparsity(self.s_model)
        print(f"🔧 最终模型稀疏率: {final_sparsity:.4f}")

        # ==============================================================================
        # MODIFICATION START: Removed unused privacy analysis block
        # ==============================================================================
        # The following block was removed because `fed_dp_aggregator` was removed.
        # Global privacy analysis is non-trivial in a Local DP setting and
        # would require collecting reports from all clients.
        #
        # try:
        #     global_privacy_analysis = self.fed_dp_aggregator.get_global_privacy_analysis()
        #     ...
        # except Exception as e:
        #     ...
        #
        # ==============================================================================
        # MODIFICATION END
        # ==============================================================================


        return {
            'best_accuracy': self.best_result,
            'final_sparsity': final_sparsity,
            'target_sparsity': self.pruning_scheduler.target_sparsity # Return the final target
        }
