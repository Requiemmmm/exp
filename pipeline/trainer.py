import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EvalPrediction,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    DataCollator,
)
from typing import Dict, List, Any, Tuple, Callable, Union, Optional, Sequence
from tqdm import tqdm
from loguru import logger

from transformers import Trainer as DefaultTrainer
from transformers.trainer import (
    unwrap_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
)

from modeling.mask import Mask
from modeling.modeling_cofi_bert import (
    CoFiBertForSequenceClassification,
)

SModel = CoFiBertForSequenceClassification


class StableLagrangianOptimizer:
    """
    稳定的拉格朗日优化器
    解决梯度爆炸和数值不稳定问题
    """

    def __init__(self, initial_lambda=0.01, max_lambda=1.0, decay_factor=0.99):
        self.lambda1 = initial_lambda
        self.lambda2 = initial_lambda
        self.max_lambda = max_lambda
        self.decay_factor = decay_factor
        self.violation_history = []
        self.step_count = 0

    '''def update_multipliers(self, violations, adaptive_lr=1e-4):
        """稳定的拉格朗日乘数更新"""

        # 记录违反历史，用于自适应调整
        self.violation_history.append(violations)
        if len(self.violation_history) > 20:
            self.violation_history = self.violation_history[-20:]

        # 计算自适应学习率
        if len(self.violation_history) > 5:
            recent_violations = np.array(self.violation_history[-5:])
            violation_variance = np.var(recent_violations, axis=0)
            # 如果违反程度变化剧烈，降低学习率
            lr_scale = 1.0 / (1.0 + violation_variance.mean())
            effective_lr = adaptive_lr * lr_scale
        else:
            effective_lr = adaptive_lr

        # 更新拉格朗日乘数，确保有界性
        self.lambda1 = np.clip(
            self.lambda1 + effective_lr * violations[0],
            0.0, self.max_lambda
        )
        self.lambda2 = np.clip(
            self.lambda2 + effective_lr * violations[1],
            0.0, self.max_lambda
        )

        # 引入衰减防止长期累积
        if self.step_count % 10 == 0:  # 每10步进行一次衰减
            self.lambda1 *= self.decay_factor
            self.lambda2 *= self.decay_factor

        self.step_count += 1

        return self.lambda1, self.lambda2'''

    # 在 pipeline/trainer.py 的 StableLagrangianOptimizer 类中

    def update_multipliers(self, violations, adaptive_lr=1e-4):
        """稳定的拉格朗日乘数更新（已修复设备不匹配问题）"""

        # --- 这是核心修改点 ---
        # 在方法开始时，立即将GPU张量转换为CPU上的标量值
        violation1_scalar = violations[0].item()  # .item() 会自动完成 .detach().cpu() 并提取数值
        violation2_scalar = violations[1]  # violations[1] 本身就是CPU上的浮点数，无需改变

        cpu_violations = [violation1_scalar, violation2_scalar]
        # ---------------------

        # 记录违反历史，用于自适应调整
        self.violation_history.append(cpu_violations)  # 确保历史记录中只包含CPU标量
        if len(self.violation_history) > 20:
            self.violation_history = self.violation_history[-20:]

        # 计算自适应学习率
        if len(self.violation_history) > 5:
            # 现在这里的np.array操作是安全的
            recent_violations = np.array(self.violation_history[-5:])
            violation_variance = np.var(recent_violations, axis=0)
            # 如果违反程度变化剧烈，降低学习率
            lr_scale = 1.0 / (1.0 + violation_variance.mean())
            effective_lr = adaptive_lr * lr_scale
        else:
            effective_lr = adaptive_lr

        # 更新拉格朗日乘数，确保有界性 (现在这里的运算也是安全的)
        self.lambda1 = np.clip(
            self.lambda1 + effective_lr * violation1_scalar,  # 使用转换后的标量值
            0.0, self.max_lambda
        )
        self.lambda2 = np.clip(
            self.lambda2 + effective_lr * violation2_scalar,  # 使用转换后的标量值
            0.0, self.max_lambda
        )

        # 引入衰减防止长期累积
        if self.step_count % 10 == 0:  # 每10步进行一次衰减
            self.lambda1 *= self.decay_factor
            self.lambda2 *= self.decay_factor

        self.step_count += 1

        return self.lambda1, self.lambda2

    def compute_lagrangian_loss(self, constraint_violations):
        """计算稳定的拉格朗日损失"""
        # 使用平滑的约束惩罚而不是线性惩罚
        smooth_penalty1 = torch.relu(constraint_violations[0]) ** 2
        smooth_penalty2 = torch.relu(constraint_violations[1]) ** 2

        lagrangian_loss = (
                self.lambda1 * smooth_penalty1 +
                self.lambda2 * smooth_penalty2
        )

        # 限制拉格朗日损失的上界，防止数值爆炸
        return torch.clamp(lagrangian_loss, max=2.0)


class BalancedLossCalculator:
    """
    平衡的多组件损失计算器
    解决损失异常高的问题
    """

    def __init__(self):
        # 基于经验的损失权重
        self.base_weights = {
            'classification': 1.0,  # 主要任务
            'distillation': 0.1,  # 知识蒸馏辅助
            'lagrangian': 1.0,  # 约束惩罚
            'regularization': 0.001  # 正则化
        }

        # 损失历史用于动态调整
        self.loss_history = {key: [] for key in self.base_weights}
        self.adaptive_weights = self.base_weights.copy()

    def compute_balanced_loss(self, loss_dict):
        """计算平衡的总损失"""

        # 更新损失历史
        for key, value in loss_dict.items():
            if key in self.loss_history and torch.isfinite(value):
                self.loss_history[key].append(value.item())
                if len(self.loss_history[key]) > 50:
                    self.loss_history[key] = self.loss_history[key][-50:]

        # 动态调整权重
        self._adjust_weights()

        # 计算加权总损失
        total_loss = 0.0
        loss_components = {}

        for component, loss_value in loss_dict.items():
            if component in self.adaptive_weights:
                # 数值稳定性检查
                if torch.isfinite(loss_value) and loss_value < 100:
                    weighted_loss = self.adaptive_weights[component] * loss_value
                    total_loss += weighted_loss
                    loss_components[component] = weighted_loss.item()
                else:
                    print(f"⚠️ 跳过异常损失 {component}: {loss_value}")
                    loss_components[component] = 0.0

        # 最终安全检查
        if not torch.isfinite(total_loss) or total_loss > 10:
            print(f"⚠️ 总损失异常 {total_loss}，回退到分类损失")
            total_loss = loss_dict.get('classification', torch.tensor(1.0))

        return total_loss, loss_components

    def _adjust_weights(self):
        """基于损失历史动态调整权重"""

        # 计算各组件损失的相对量级
        loss_scales = {}
        for component, history in self.loss_history.items():
            if len(history) > 5:
                recent_mean = np.mean(history[-10:])
                loss_scales[component] = recent_mean

        if len(loss_scales) > 1:
            # 使用分类损失作为基准
            base_scale = loss_scales.get('classification', 1.0)

            for component in self.adaptive_weights:
                if component in loss_scales and component != 'classification':
                    # 调整权重使各组件损失量级相当
                    scale_ratio = loss_scales[component] / base_scale
                    if scale_ratio > 1:
                        self.adaptive_weights[component] = self.base_weights[component] / scale_ratio
                    else:
                        self.adaptive_weights[component] = self.base_weights[component]


class AdaptiveGradientClipper:
    """
    自适应梯度裁剪器
    解决梯度爆炸问题
    """

    def __init__(self, initial_clip_norm=1.0, percentile=75.0, warmup_steps=100):
        self.clip_norm = initial_clip_norm
        self.percentile = percentile
        self.warmup_steps = warmup_steps
        self.grad_norm_history = []
        self.clipped_count = 0
        self.total_count = 0

    def clip_gradients(self, model, adaptive=True):
        """
        执行自适应梯度裁剪
        返回: (裁剪前范数, 裁剪后范数, 是否被裁剪)
        """
        # 计算当前梯度范数
        total_norm = 0.0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                # 🚨 强制限制单个参数梯度，防止数值爆炸
                param.grad.data.clamp_(-100, 100)
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count == 0:
            return 0.0, 0.0, False

        # 归一化梯度范数
        total_norm = (total_norm ** 0.5) / max(1, param_count ** 0.5)

        # 记录历史用于自适应调整
        self.grad_norm_history.append(min(total_norm, 1000))  # 限制记录的最大值
        if len(self.grad_norm_history) > 1000:
            self.grad_norm_history = self.grad_norm_history[-1000:]

        # 自适应调整裁剪阈值
        if adaptive and len(self.grad_norm_history) > self.warmup_steps:
            self.clip_norm = min(
                np.percentile(self.grad_norm_history, self.percentile),
                10.0  # 上界限制
            )

        # 🚨 强制限制梯度范数上界
        max_allowed_norm = min(self.clip_norm, 10.0)
        was_clipped = total_norm > max_allowed_norm
        clipped_norm = total_norm

        if was_clipped:
            clip_coef = max_allowed_norm / (total_norm + 1e-6)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
            clipped_norm = max_allowed_norm
            self.clipped_count += 1

        self.total_count += 1

        return total_norm, clipped_norm, was_clipped

    def get_clipping_stats(self):
        """获取裁剪统计信息"""
        return {
            'clipping_ratio': self.clipped_count / max(1, self.total_count),
            'current_clip_norm': self.clip_norm,
            'avg_grad_norm': np.mean(self.grad_norm_history[-100:]) if self.grad_norm_history else 0,
            'grad_norm_std': np.std(self.grad_norm_history[-100:]) if len(self.grad_norm_history) > 1 else 0
        }


class DistillTrainer(DefaultTrainer):

    def __init__(self,
                 s_model: Union[PreTrainedModel, nn.Module] = None,
                 t_model: Union[PreTrainedModel, nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 model_init: Optional[Callable[[], PreTrainedModel]] = None,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
                 ):
        assert callbacks is None
        super().__init__(
            s_model, args, data_collator, train_dataset, eval_dataset,
            tokenizer, model_init, compute_metrics, callbacks, optimizers,
            preprocess_logits_for_metrics
        )
        self.t_model = t_model
        device = next(self.model.parameters()).device
        self.t_model.to(device)
        self.t_model.eval()

        self.distill_switch = False
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        # self.mse_loss = nn.MSELoss() # REMOVED: No longer needed for hidden layer distillation

        self.start_sparsity = 1.
        self.target_sparsity = self.args.target_sparsity

        self.reg_params = []

        self.per_layer_mask_groups: List[Tuple[Mask, ...]] = []
        self.init_reg_params()
        self.ffn_masks: List[Mask] = []
        self.init_ffn_masks()

        # ✅ 添加稳定性组件
        self.gradient_clipper = AdaptiveGradientClipper(
            initial_clip_norm=getattr(args, 'max_grad_norm', 1.0),
            percentile=75.0,
            warmup_steps=50
        )

        self.lagrangian_optimizer = StableLagrangianOptimizer(
            initial_lambda=0.001,  # 更小的初始值
            max_lambda=0.5,  # 更严格的上界
            decay_factor=0.995  # 轻微衰减
        )

        self.loss_calculator = BalancedLossCalculator()

        # ✅ 添加训练统计追踪
        self.training_stats = {
            'sparsity_history': [],
            'distill_weight_history': [],
            'grad_norm_history': [],
            'loss_components': {'total': [], 'classification': [], 'distillation': [], 'lagrangian': []},
            'lambda_history': {'lambda1': [], 'lambda2': []}
        }

        print(f"✅ DistillTrainer初始化完成")
        print(f"   - 目标稀疏率: {self.target_sparsity}")
        print(f"   - 梯度裁剪: 启用自适应裁剪")
        print(f"   - 拉格朗日优化: 稳定版本")
        print(f"   - 损失平衡: 启用")

    def init_ffn_masks(self):
        model: SModel = self.model
        for layer in model.bert.encoder.layer:
            FFN_mask = layer.output.mask
            self.ffn_masks.append(FFN_mask)

    def init_reg_params(self):
        for name, _ in self.model.named_parameters():
            if name.endswith('reg_lambda_1') or \
                    name.endswith('reg_lambda_2') or \
                    name.endswith('log_alpha'):
                self.reg_params.append(name)
        model: SModel = self.model

        for layer in model.bert.encoder.layer:
            head_mask = layer.attention.self.mask
            filter_mask = layer.output.dense.mask
            MHA_mask = layer.attention.output.mask
            FFN_mask = layer.output.mask

            self.per_layer_mask_groups.append((
                head_mask,
                MHA_mask,
                FFN_mask,
                filter_mask,
            ))

    def create_optimizer(self):
        """
        Setup the optimizer.
        增强版优化器设置，使用更保守的学习率
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n in decay_parameters and p.requires_grad and n not in self.reg_params)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n not in decay_parameters and p.requires_grad and n not in self.reg_params)
                    ],
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in self.reg_params and "reg" not in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.reg_learning_rate,  # 🚨 限制正则化学习率上界
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in self.reg_params and "reg" in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": -min(self.args.reg_learning_rate, 1e-6),  # 🚨 限制正则化学习率上界
                }
            ]

            optimizer_cls, optimizer_kwargs = DefaultTrainer.get_optimizer_cls_and_kwargs(self.args)

            # 🚨 限制主学习率
            if 'lr' in optimizer_kwargs:
                optimizer_kwargs['lr'] = min(optimizer_kwargs['lr'], 5e-5)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def train(self,
              resume_from_checkpoint: Optional[Union[str, bool]] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None,
              ignore_keys_for_eval: Optional[List[str]] = None,
              **kwargs
              ):
        self.distill_switch = True
        print(f"🚀 开始蒸馏训练...")
        result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        self.distill_switch = False

        print(f"✅ 蒸馏训练完成")
        self._print_training_summary()

        return result

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        增强的训练步骤，集成梯度裁剪和统计追踪
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # 标准反向传播（移除apex依赖）
        if hasattr(self, 'accelerator') and self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        # ✅ 应用自适应梯度裁剪
        original_norm, clipped_norm, was_clipped = self.gradient_clipper.clip_gradients(
            model, adaptive=True
        )

        # 记录梯度统计
        self.training_stats['grad_norm_history'].append({
            'original_norm': original_norm,
            'clipped_norm': clipped_norm,
            'was_clipped': was_clipped
        })

        return loss.detach() / self.args.gradient_accumulation_steps

    def _compute_differentiable_sparsity_ratio(self):
        """计算一个可微分的、基于所有mask的L0范数期望值的稀疏率代理"""
        total_L0_norm = 0
        total_mask_params = 0
        # 遍历所有层的mask组
        for mask_group in self.per_layer_mask_groups:
            for mask in mask_group:
                # L() 返回的是期望保留的参数量（或维度）
                total_L0_norm += mask.L().sum()
                total_mask_params += mask.features

        if total_mask_params == 0:
            return 0.0

        # 返回期望的“保留率”，即 1 - 稀疏率
        expected_retention_ratio = total_L0_norm / total_mask_params
        return expected_retention_ratio

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        ✅ 修复后的损失计算方法
        解决梯度爆炸和损失异常高的问题
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # MODIFIED: output_hidden_states is still needed for the s_logits, but not for t_hidden_states
        if "output_hidden_states" in inputs:
            inputs["output_hidden_states"] = inputs["output_hidden_states"] or self.distill_switch
        else:
            inputs["output_hidden_states"] = self.distill_switch

        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # 分类损失
        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                classification_loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                classification_loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            classification_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # 损失字典
        loss_dict = {'classification': classification_loss}

        # ✅ 安全的蒸馏损失
        if self.distill_switch:
            try:
                distill_loss = self.compute_adaptive_distill_loss(
                    unwrap_model(model),
                    inputs,
                    outputs["logits"],
                    # s_hidden_states is no longer needed as an argument
                )
                if torch.isfinite(distill_loss) and distill_loss < 10:
                    loss_dict['distillation'] = distill_loss
                else:
                    print(f"⚠️ 蒸馏损失异常: {distill_loss}，跳过")
            except Exception as e:
                print(f"⚠️ 蒸馏损失计算失败: {e}")

        # ✅ 稳定的拉格朗日损失
        '''if self.distill_switch:
            try:
                # 计算当前模型稀疏率 - MODIFICATION: This is where we use the unified standard
                current_sparsity = self._compute_current_sparsity(unwrap_model(model))
                target_sparsity = self.target_sparsity

                # 计算约束违反
                sparsity_violation = current_sparsity - target_sparsity
                violations = [sparsity_violation, 0.0]

                # 更新拉格朗日乘数
                lambda1, lambda2 = self.lagrangian_optimizer.update_multipliers(violations)

                # 计算拉格朗日损失
                lagrangian_loss = self.lagrangian_optimizer.compute_lagrangian_loss(
                    [torch.tensor(sparsity_violation, device=classification_loss.device),
                     torch.tensor(0.0, device=classification_loss.device)]
                )

                if torch.isfinite(lagrangian_loss):
                    loss_dict['lagrangian'] = lagrangian_loss

                # 记录统计信息
                self.training_stats['lambda_history']['lambda1'].append(lambda1)
                self.training_stats['lambda_history']['lambda2'].append(lambda2)

            except Exception as e:
                print(f"⚠️ 拉格朗日损失计算失败: {e}")'''
        if self.distill_switch:
            try:
                # --- 这是核心修改点 ---
                # 使用可微分的稀疏率代理来计算约束，以确保梯度流
                differentiable_retention = self._compute_differentiable_sparsity_ratio()
                # 目标保留率 = 1 - 目标稀疏率
                target_retention = 1.0 - self.target_sparsity
                sparsity_violation = differentiable_retention - target_retention

                # 保留实际稀疏率用于日志监控
                current_sparsity_for_log = self._compute_current_sparsity(unwrap_model(model))
                # ---------------------

                violations = [sparsity_violation, 0.0]

                # 更新拉格朗日乘数
                lambda1, lambda2 = self.lagrangian_optimizer.update_multipliers(violations)

                # 计算拉格朗日损失
                lagrangian_loss = self.lagrangian_optimizer.compute_lagrangian_loss(
                    [torch.tensor(sparsity_violation, device=classification_loss.device),
                     torch.tensor(0.0, device=classification_loss.device)]
                )

                if torch.isfinite(lagrangian_loss):
                    loss_dict['lagrangian'] = lagrangian_loss

                # 记录统计信息 (使用实际稀疏率)
                self.training_stats['sparsity_history'].append(current_sparsity_for_log)
                self.training_stats['lambda_history']['lambda1'].append(lambda1)
                self.training_stats['lambda_history']['lambda2'].append(lambda2)

            except Exception as e:
                print(f"⚠️ 拉格朗日损失计算失败: {e}")

        # ✅ 计算平衡的总损失
        total_loss, loss_components = self.loss_calculator.compute_balanced_loss(loss_dict)

        # ✅ 记录损失组件用于分析
        for component, value in loss_components.items():
            if component in self.training_stats['loss_components']:
                self.training_stats['loss_components'][component].append(value)

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_current_sparsity(self, model):
        """计算模型当前的实际稀疏率"""
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
                zero_params += (param.abs() < 1e-8).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0

    def mask_select(self,
                    value: torch.Tensor,
                    mask: torch.Tensor
                    ) -> torch.Tensor:
        assert value.shape[:-1] == mask.shape
        D = value.shape[-1]
        value = value.view(-1, D)
        mask = mask.view(-1).bool()
        return value[mask]

    def compute_adaptive_distill_loss(self,
                                      model: SModel,
                                      inputs: Dict,
                                      s_logits: torch.Tensor,
                                      # s_hidden_states: torch.Tensor, # REMOVED: No longer needed
                                      ):
        """
        MODIFIED: 自适应蒸馏损失计算 (简化版)
        - 移除对隐藏层的特征蒸馏，只保留对logits的蒸馏
        - 极大提升训练稳定性和速度
        """
        with torch.no_grad():
            # Set output_hidden_states to False for teacher model to save computation
            inputs_for_teacher = inputs.copy()
            inputs_for_teacher["output_hidden_states"] = False
            t_outputs = self.t_model(**inputs_for_teacher)
            t_logits = t_outputs["logits"]
            # t_hidden_states no longer fetched

        T = self.args.distill_T

        # ✅ 计算当前稀疏率并获取自适应蒸馏权重
        current_sparsity = self._compute_current_sparsity(model)
        adaptive_distill_lambda = self.get_adaptive_distill_weight(current_sparsity)

        # 记录当前参数用于分析
        self.training_stats['sparsity_history'].append(current_sparsity)
        self.training_stats['distill_weight_history'].append(adaptive_distill_lambda)

        # 预测蒸馏损失 (Logits-based KL Divergence)
        pred_loss = self.kl_loss(
            torch.log_softmax(s_logits / T, dim=-1),
            torch.log_softmax(t_logits / T, dim=-1),
        ) * (T ** 2)

        # =================================================================================
        # MODIFICATION START: Removed all hidden layer distillation logic for stability
        # =================================================================================

        # The following block has been removed:
        # assert len(t_hidden_states) == len(s_hidden_states)
        # proj = model.bert.distill_projection
        # t_hidden_states = [self.mask_select(t_h, mask) for t_h in t_hidden_states]
        # s_hidden_states = [proj(self.mask_select(s_h, mask)) for s_h in s_hidden_states]
        # ... layer matching algorithm ...
        # ... feature_weight calculation ...
        # ... _layer_loss calculation ...
        # ... total_layer_loss calculation ...

        # ✅ 使用简化的损失
        distill_loss = adaptive_distill_lambda * pred_loss

        # =================================================================================
        # MODIFICATION END
        # =================================================================================

        return distill_loss

    def get_adaptive_distill_weight(self, current_sparsity):
        """
        获取自适应蒸馏权重
        使用更温和的权重调整
        """
        base_weight = 0.05  # 降低基础权重
        # 温和增长：稀疏率0.0->0.3时，蒸馏权重0.05->0.2
        adaptive_weight = base_weight + 0.15 * current_sparsity
        return min(adaptive_weight, 0.3)  # 限制最大权重

    def compute_target_sparsity(self):
        return self.target_sparsity

    def compute_lagrangian_loss(self):
        """传统的拉格朗日损失计算（保留兼容性, 用于评估）"""
        # MODIFICATION: Use actual sparsity for this calculation too for consistency
        s = self._compute_current_sparsity(unwrap_model(self.model))
        t = self.compute_target_sparsity()

        # These lambdas are part of the model parameters for a different optimization method
        # which might not be actively used if StableLagrangianOptimizer is in effect.
        # We keep this for evaluation purposes as requested by original structure.
        lambda_1 = self.model.bert.reg_lambda_1
        lambda_2 = self.model.bert.reg_lambda_2
        lagrangian_loss = lambda_1 * (s - t).abs() + lambda_2 * torch.pow(s - t, 2.)
        return lagrangian_loss

    def compute_sparsity(self):
        """
        ✅ 理论稀疏性计算
        返回基于mask参数的理论稀疏率
        NOTE: This method is now considered deprecated for control and primary evaluation,
        but kept for potential diagnostic purposes. The primary metric is _compute_current_sparsity.
        """
        num_layers = 12
        num_heads = 12
        hidden_size = 768
        ffn_size = 768 * 4

        # 计算总参数数量
        total_params = (hidden_size * hidden_size * 4 + hidden_size * ffn_size * 2) * num_layers

        # 计算剩余参数数量
        remaining_params = []
        hidden_mask = torch.ones([768]).cuda()

        for mask_group in self.per_layer_mask_groups:
            head_mask, MHA_mask, FFN_mask, filter_mask = mask_group

            MHA_mask_L = MHA_mask.L()
            head_mask_L = head_mask.L()
            FFN_mask_L = FFN_mask.L()

            # 注意力层参数
            attention_params = 4 * 64 * hidden_mask.sum() * head_mask_L.sum() * MHA_mask_L.sum()
            remaining_params.append(attention_params)

            # FFN层参数
            mask = torch.outer(hidden_mask, filter_mask.L())
            mask = mask * FFN_mask_L
            ffn_params = 2 * mask.sum()
            remaining_params.append(ffn_params)

        total_remaining = torch.stack(remaining_params).sum()

        sparsity = 1.0 - (total_remaining / total_params)

        return sparsity

    def evaluate(self,
                 eval_dataset: Optional[Dataset] = None,
                 ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval"
                 ) -> Dict[str, float]:
        """
        增强的评估方法，包含详细的性能和训练分析
        """
        if self.args.local_rank == 0:
            with torch.no_grad():
                # 获取拉格朗日乘数
                lambda_1 = getattr(self.model.bert, 'reg_lambda_1', torch.tensor(0.0))
                lambda_2 = getattr(self.model.bert, 'reg_lambda_2', torch.tensor(0.0))

                if torch.is_tensor(lambda_1):
                    lambda_1_val = lambda_1.item()
                else:
                    lambda_1_val = lambda_1

                if torch.is_tensor(lambda_2):
                    lambda_2_val = lambda_2.item()
                else:
                    lambda_2_val = lambda_2

                # MODIFICATION: Unify sparsity reporting
                actual_sparsity = self._compute_current_sparsity(unwrap_model(self.model))
                t_sparsity = self.compute_target_sparsity()

                # 计算拉格朗日损失
                try:
                    lagrangian_loss = self.compute_lagrangian_loss()
                    lagrangian_val = lagrangian_loss.item() if torch.is_tensor(lagrangian_loss) else lagrangian_loss
                except:
                    lagrangian_val = 0.0

                print(f"📊 训练统计:")
                print(f"   λ₁ (model param): {lambda_1_val:.6f}")
                print(f"   λ₂ (model param): {lambda_2_val:.6f}")
                # print(f"   系统稀疏率: {sparsity:.4f}") # REMOVED: Deprecating theoretical sparsity
                print(f"   实际稀疏率 (Actual Sparsity): {actual_sparsity:.4f}")
                print(f"   目标稀疏率: {t_sparsity:.4f}")
                print(f"   拉格朗日损失 (eval): {lagrangian_val:.6f}")

                # 打印梯度裁剪统计
                clip_stats = self.gradient_clipper.get_clipping_stats()
                print(f"   梯度裁剪率: {clip_stats['clipping_ratio']:.3f}")
                print(f"   当前裁剪阈值: {clip_stats['current_clip_norm']:.3f}")
                print(f"   平均梯度范数: {clip_stats['avg_grad_norm']:.3f}")

                # 打印拉格朗日乘数历史
                if self.training_stats['lambda_history']['lambda1']:
                    recent_lambda1 = np.mean(self.training_stats['lambda_history']['lambda1'][-10:])
                    recent_lambda2 = np.mean(self.training_stats['lambda_history']['lambda2'][-10:])
                    print(f"   稳定λ₁ (trainer): {recent_lambda1:.6f}")
                    print(f"   稳定λ₂ (trainer): {recent_lambda2:.6f}")

        past_distill_switch = self.distill_switch
        self.distill_switch = False
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.distill_switch = past_distill_switch

        with torch.no_grad():
            # MODIFICATION: Unify sparsity reporting in results dict
            # results['sparsity'] = self.compute_sparsity() # REMOVED
            results['actual_sparsity'] = self._compute_current_sparsity(unwrap_model(self.model))
            results['target_sparsity'] = self.compute_target_sparsity()

            try:
                results['lagrangian_loss'] = self.compute_lagrangian_loss().item()
            except:
                results['lagrangian_loss'] = 0.0

            # 添加蒸馏权重信息
            current_sparsity = results['actual_sparsity']
            adaptive_distill_weight = self.get_adaptive_distill_weight(current_sparsity)
            results['distill_weight'] = adaptive_distill_weight

            # 添加训练稳定性指标
            if self.training_stats['grad_norm_history']:
                recent_grads = self.training_stats['grad_norm_history'][-10:]
                avg_grad_norm = np.mean([g['original_norm'] for g in recent_grads])
                results['avg_grad_norm'] = avg_grad_norm

        return results

    def _print_training_summary(self):
        """打印训练总结"""
        if not self.training_stats['sparsity_history']:
            return

        print(f"\n📈 训练总结:")

        if self.training_stats['sparsity_history']:
            final_sparsity = self.training_stats['sparsity_history'][-1]
            print(f"   最终稀疏率: {final_sparsity:.4f}")

        if self.training_stats['distill_weight_history']:
            final_distill_weight = self.training_stats['distill_weight_history'][-1]
            print(f"   最终蒸馏权重: {final_distill_weight:.4f}")

        if self.training_stats['grad_norm_history']:
            recent_grads = self.training_stats['grad_norm_history'][-10:]
            avg_original = np.mean([g['original_norm'] for g in recent_grads])
            clip_ratio = np.mean([g['was_clipped'] for g in recent_grads])
            print(f"   平均梯度范数: {avg_original:.4f}")
            print(f"   最近裁剪率: {clip_ratio:.3f}")

        # 损失组件分析
        for component in ['total', 'classification', 'distillation', 'lagrangian']:
            if component in self.training_stats['loss_components'] and self.training_stats['loss_components'][
                component]:
                recent_losses = self.training_stats['loss_components'][component][-10:]
                avg_loss = np.mean(recent_losses)
                print(f"   平均{component}损失: {avg_loss:.6f}")

        # 拉格朗日乘数稳定性
        if self.training_stats['lambda_history']['lambda1']:
            lambda1_stability = np.std(self.training_stats['lambda_history']['lambda1'][-20:])
            lambda2_stability = np.std(self.training_stats['lambda_history']['lambda2'][-20:])
            print(f"   λ₁稳定性(std): {lambda1_stability:.6f}")
            print(f"   λ₂稳定性(std): {lambda2_stability:.6f}")

    def save_training_stats(self, output_dir: str):
        """保存训练统计信息"""
        if not hasattr(self, 'training_stats'):
            return

        import json
        import os

        stats_file = os.path.join(output_dir, 'training_stats.json')

        # 转换tensor为普通数值
        stats_to_save = {}
        for key, value in self.training_stats.items():
            if isinstance(value, dict):
                stats_to_save[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        stats_to_save[key][subkey] = [
                            v.item() if torch.is_tensor(v) else v
                            for v in subvalue
                        ]
                    else:
                        stats_to_save[key][subkey] = subvalue
            elif isinstance(value, list):
                stats_to_save[key] = [
                    v.item() if torch.is_tensor(v) else v
                    for v in value
                ]
            else:
                stats_to_save[key] = value

        # 添加梯度裁剪统计
        stats_to_save['gradient_clipping_stats'] = self.gradient_clipper.get_clipping_stats()

        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)

        print(f"📁 训练统计已保存到: {stats_file}")

    def get_training_diagnostics(self):
        """获取训练诊断信息"""
        diagnostics = {
            'gradient_health': {
                'avg_norm': np.mean([g['original_norm'] for g in self.training_stats['grad_norm_history'][-10:]]) if
                self.training_stats['grad_norm_history'] else 0,
                'clipping_rate': self.gradient_clipper.get_clipping_stats()['clipping_ratio'],
                'stability': 'good' if self.gradient_clipper.get_clipping_stats()[
                                           'clipping_ratio'] < 0.5 else 'concerning'
            },
            'loss_health': {
                'classification_stable': len(self.training_stats['loss_components']['classification']) > 5 and
                                         np.std(self.training_stats['loss_components']['classification'][-10:]) < 1.0,
                'total_loss_trend': 'decreasing' if len(self.training_stats['loss_components']['total']) > 5 and
                                                    self.training_stats['loss_components']['total'][-1] <
                                                    self.training_stats['loss_components']['total'][-5]
                else 'stable_or_increasing'
            },
            'sparsity_health': {
                'target_vs_actual': abs(self.training_stats['sparsity_history'][-1] - self.target_sparsity) if
                self.training_stats['sparsity_history'] else float('inf'),
                'sparsity_progression': 'healthy' if self.training_stats['sparsity_history'] and
                                                     len(set(self.training_stats['sparsity_history'][
                                                             -5:])) > 1 else 'stagnant'
            },
            'lagrangian_health': {
                'lambda_stability': np.std(self.training_stats['lambda_history']['lambda1'][-10:]) if len(
                    self.training_stats['lambda_history']['lambda1']) > 10 else float('inf'),
                'convergence': 'converging' if len(self.training_stats['lambda_history']['lambda1']) > 10 and
                                               np.std(self.training_stats['lambda_history']['lambda1'][
                                                      -10:]) < 0.1 else 'diverging'
            }
        }

        return diagnostics

    def emergency_reset(self):
        """紧急重置：在训练不稳定时调用"""
        print("🚨 执行紧急重置...")

        # 重置拉格朗日乘数
        self.lagrangian_optimizer = StableLagrangianOptimizer(
            initial_lambda=0.001,
            max_lambda=0.1,  # 更严格的限制
            decay_factor=0.99
        )

        # 重置梯度裁剪器
        self.gradient_clipper = AdaptiveGradientClipper(
            initial_clip_norm=0.5,  # 更严格的初始裁剪
            percentile=50.0,
            warmup_steps=20
        )

        # 清空历史统计
        for key in self.training_stats:
            if isinstance(self.training_stats[key], list):
                self.training_stats[key] = []
            elif isinstance(self.training_stats[key], dict):
                for subkey in self.training_stats[key]:
                    if isinstance(self.training_stats[key][subkey], list):
                        self.training_stats[key][subkey] = []

        # 强制垃圾回收
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        print("✅ 紧急重置完成")
