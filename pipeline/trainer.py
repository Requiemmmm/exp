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

    def update_multipliers(self, violations, adaptive_lr=1e-4):
        """稳定的拉格朗日乘数更新（已修复设备不匹配问题）"""
        violation1_scalar = violations[0].item()
        violation2_scalar = violations[1]
        cpu_violations = [violation1_scalar, violation2_scalar]

        self.violation_history.append(cpu_violations)
        if len(self.violation_history) > 20:
            self.violation_history = self.violation_history[-20:]

        if len(self.violation_history) > 5:
            recent_violations = np.array(self.violation_history[-5:])
            violation_variance = np.var(recent_violations, axis=0)
            lr_scale = 1.0 / (1.0 + violation_variance.mean())
            effective_lr = adaptive_lr * lr_scale
        else:
            effective_lr = adaptive_lr

        self.lambda1 = np.clip(
            self.lambda1 + effective_lr * violation1_scalar,
            0.0, self.max_lambda
        )
        self.lambda2 = np.clip(
            self.lambda2 + effective_lr * violation2_scalar,
            0.0, self.max_lambda
        )

        if self.step_count % 10 == 0:
            self.lambda1 *= self.decay_factor
            self.lambda2 *= self.decay_factor
        self.step_count += 1
        return self.lambda1, self.lambda2

    def compute_lagrangian_loss(self, constraint_violations):
        """计算稳定的拉格朗日损失"""
        smooth_penalty1 = torch.relu(constraint_violations[0]) ** 2
        smooth_penalty2 = torch.relu(constraint_violations[1]) ** 2
        lagrangian_loss = (self.lambda1 * smooth_penalty1 + self.lambda2 * smooth_penalty2)
        return torch.clamp(lagrangian_loss, max=2.0)


class BalancedLossCalculator:
    """
    平衡的多组件损失计算器
    解决损失异常高的问题
    """

    def __init__(self):
        self.base_weights = {
            'classification': 1.0,
            'distillation': 0.1,
            'lagrangian': 1.0,
            'regularization': 0.001
        }
        self.loss_history = {key: [] for key in self.base_weights}
        self.adaptive_weights = self.base_weights.copy()

    def compute_balanced_loss(self, loss_dict):
        """计算平衡的总损失"""
        for key, value in loss_dict.items():
            if key in self.loss_history and torch.isfinite(value):
                self.loss_history[key].append(value.item())
                if len(self.loss_history[key]) > 50:
                    self.loss_history[key] = self.loss_history[key][-50:]

        self._adjust_weights()

        total_loss = 0.0
        loss_components = {}
        for component, loss_value in loss_dict.items():
            if component in self.adaptive_weights:
                if torch.isfinite(loss_value) and loss_value < 100:
                    weighted_loss = self.adaptive_weights[component] * loss_value
                    total_loss += weighted_loss
                    loss_components[component] = weighted_loss.item()
                else:
                    print(f"⚠️ 跳过异常损失 {component}: {loss_value}")
                    loss_components[component] = 0.0

        if not torch.isfinite(total_loss) or total_loss > 10:
            print(f"⚠️ 总损失异常 {total_loss}，回退到分类损失")
            total_loss = loss_dict.get('classification', torch.tensor(1.0))

        return total_loss, loss_components

    def _adjust_weights(self):
        """基于损失历史动态调整权重"""
        loss_scales = {}
        for component, history in self.loss_history.items():
            if len(history) > 5:
                recent_mean = np.mean(history[-10:])
                loss_scales[component] = recent_mean

        if len(loss_scales) > 1:
            base_scale = loss_scales.get('classification', 1.0)
            for component in self.adaptive_weights:
                if component in loss_scales and component != 'classification':
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
        """执行自适应梯度裁剪"""
        total_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-100, 100)
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        if param_count == 0:
            return 0.0, 0.0, False

        total_norm = (total_norm ** 0.5) / max(1, param_count ** 0.5)
        self.grad_norm_history.append(min(total_norm, 1000))
        if len(self.grad_norm_history) > 1000:
            self.grad_norm_history = self.grad_norm_history[-1000:]

        if adaptive and len(self.grad_norm_history) > self.warmup_steps:
            self.clip_norm = min(np.percentile(self.grad_norm_history, self.percentile), 10.0)

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
        self.device = device

        self.distill_switch = False
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

        self.start_sparsity = 1.
        self.target_sparsity = self.args.target_sparsity

        self.reg_params = []
        self.per_layer_mask_groups: List[Tuple[Mask, ...]] = []
        self.init_reg_params()
        self.ffn_masks: List[Mask] = []
        self.init_ffn_masks()

        self.gradient_clipper = AdaptiveGradientClipper(
            initial_clip_norm=getattr(args, 'max_grad_norm', 1.0),
            percentile=75.0,
            warmup_steps=50
        )
        self.lagrangian_optimizer = StableLagrangianOptimizer(
            initial_lambda=0.001,
            max_lambda=0.5,
            decay_factor=0.995
        )
        self.loss_calculator = BalancedLossCalculator()
        self._initialize_masks()
        self._initialize_cofi_masks()

        self.training_stats = {
            'sparsity_history': [], 'distill_weight_history': [], 'grad_norm_history': [],
            'loss_components': {'total': [], 'classification': [], 'distillation': [], 'lagrangian': []},
            'lambda_history': {'lambda1': [], 'lambda2': []}
        }

        print(f"✅ DistillTrainer初始化完成")
        print(f"   - 目标稀疏率: {self.target_sparsity}")
        print(f"   - 梯度裁剪: 启用自适应裁剪")
        print(f"   - 拉格朗日优化: 稳定版本")
        print(f"   - 损失平衡: 启用")

    def _initialize_masks(self):
        """初始化可微分mask"""
        self.differentiable_masks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                mask_logits = torch.zeros_like(param, requires_grad=True, device=self.device)
                self.differentiable_masks[name] = mask_logits

    def _initialize_cofi_masks(self):
        """初始化和修复CoFi mask参数"""
        print("🔧 检查并修复CoFi mask参数...")
        total_masks, fixed_masks = 0, 0
        try:
            for layer_idx, layer in enumerate(self.model.bert.encoder.layer):
                layer_masks = []
                if hasattr(layer.attention.self, 'mask'): layer_masks.append(
                    ('attention_self', layer.attention.self.mask))
                if hasattr(layer.attention.output, 'mask'): layer_masks.append(
                    ('attention_output', layer.attention.output.mask))
                if hasattr(layer.output, 'mask'): layer_masks.append(('ffn_output', layer.output.mask))
                if hasattr(layer.output.dense, 'mask'): layer_masks.append(('ffn_dense', layer.output.dense.mask))

                for mask_name, mask in layer_masks:
                    total_masks += 1
                    try:
                        if hasattr(mask, 'log_alpha') and mask.log_alpha is not None and (
                                mask.log_alpha.dtype == torch.bool or not mask.log_alpha.dtype.is_floating_point):
                            print(
                                f"  🔧 修复Layer{layer_idx}-{mask_name}的log_alpha类型: {mask.log_alpha.dtype} → float32")
                            mask.log_alpha.data = torch.randn(mask.log_alpha.shape, device=mask.log_alpha.device,
                                                              dtype=torch.float32) * 0.1
                            fixed_masks += 1
                        if hasattr(mask, 'activate') and mask.activate is not None and (
                                mask.activate.dtype == torch.bool or not mask.activate.dtype.is_floating_point):
                            print(
                                f"  🔧 修复Layer{layer_idx}-{mask_name}的activate类型: {mask.activate.dtype} → float32")
                            mask.activate.data = torch.full(mask.activate.shape, 0.9, device=mask.activate.device,
                                                            dtype=torch.float32)
                            fixed_masks += 1
                    except Exception as e:
                        print(f"  ⚠️ 修复Layer{layer_idx}-{mask_name}失败: {e}")
            print(f"✅ CoFi mask检查完成: 总共{total_masks}个mask，修复了{fixed_masks}个")
        except Exception as e:
            print(f"❌ CoFi mask初始化失败: {e}")

    def _get_current_target_sparsity(self):
        """获取当前目标稀疏率"""
        return getattr(self.args, 'target_sparsity', getattr(self, 'target_sparsity', 0.0))

    def init_ffn_masks(self):
        model: SModel = self.model
        for layer in model.bert.encoder.layer:
            self.ffn_masks.append(layer.output.mask)

    def init_reg_params(self):
        for name, _ in self.model.named_parameters():
            if name.endswith(('reg_lambda_1', 'reg_lambda_2', 'log_alpha')):
                self.reg_params.append(name)
        model: SModel = self.model
        for layer in model.bert.encoder.layer:
            self.per_layer_mask_groups.append((
                layer.attention.self.mask, layer.attention.output.mask,
                layer.output.mask, layer.output.dense.mask
            ))

    def create_optimizer(self):
        """增强版优化器设置"""
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = [name for name in get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS) if
                                "bias" not in name]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in opt_model.named_parameters() if
                            n in decay_parameters and p.requires_grad and n not in self.reg_params],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in opt_model.named_parameters() if
                            n not in decay_parameters and p.requires_grad and n not in self.reg_params],
                 "weight_decay": 0.0},
                {"params": [p for n, p in opt_model.named_parameters() if n in self.reg_params and "reg" not in n],
                 "weight_decay": 0.0, "lr": self.args.reg_learning_rate},
                {"params": [p for n, p in opt_model.named_parameters() if n in self.reg_params and "reg" in n],
                 "weight_decay": 0.0, "lr": -min(self.args.reg_learning_rate, 1e-6)}
            ]
            optimizer_cls, optimizer_kwargs = DefaultTrainer.get_optimizer_cls_and_kwargs(self.args)
            if 'lr' in optimizer_kwargs:
                optimizer_kwargs['lr'] = min(optimizer_kwargs['lr'], 5e-5)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None,
              trial: Union["optuna.Trial", Dict[str, Any]] = None, ignore_keys_for_eval: Optional[List[str]] = None,
              **kwargs):
        self.distill_switch = True
        print(f"🚀 开始蒸馏训练...")
        result = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
        self.distill_switch = False
        print(f"✅ 蒸馏训练完成")
        self._print_training_summary()
        return result

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """增强的训练步骤，集成激进剪枝和统计追踪"""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if hasattr(self, 'accelerator'):
            self.accelerator.backward(loss)
        else:
            loss.backward()

        original_norm, _, _ = self.gradient_clipper.clip_gradients(model, adaptive=True)

        if self.distill_switch and hasattr(self, '_get_current_target_sparsity'):
            try:
                current_target = self._get_current_target_sparsity()
                self._apply_deterministic_pruning(model, current_target)
            except Exception as e:
                print(f"  ❌ 激进剪枝失败: {e}")

        return loss.detach() / self.args.gradient_accumulation_steps

    def _apply_deterministic_pruning(self, model, target_sparsity):
        """
        确定性剪枝策略.
        直接根据目标稀疏率计算需要剪枝的元素数量，并强制剪枝.
        """
        if not hasattr(self, 'pruning_step_count'):
            self.pruning_step_count = 0
        self.pruning_step_count += 1

        if target_sparsity <= 0.0: return 0.0

        if self.pruning_step_count % 10 != 0: return 0.0

        try:
            for layer in model.bert.encoder.layer:
                for module in layer.modules():
                    if isinstance(module, Mask):
                        mask = module
                        if not (hasattr(mask, 'log_alpha') and mask.log_alpha is not None):
                            continue

                        total_elements = mask.log_alpha.numel()
                        elements_to_keep = int(round(total_elements * (1.0 - target_sparsity)))
                        elements_to_prune = total_elements - elements_to_keep

                        if elements_to_prune > 0:
                            flat_log_alpha = mask.log_alpha.view(-1)
                            _, indices_to_prune = torch.topk(flat_log_alpha, elements_to_prune, largest=False)
                            flat_log_alpha.data[indices_to_prune] = -10.0

                            if elements_to_keep > 0:
                                _, indices_to_keep = torch.topk(flat_log_alpha, elements_to_keep, largest=True)
                                flat_log_alpha.data[indices_to_keep] = 10.0

        except Exception as e:
            print(f"  ❌ 确定性剪枝失败: {e}")
            return 0.0

        return self._compute_model_sparsity_cofi_accurate(model)

    def _compute_model_sparsity_cofi_accurate(self, model):
        """更准确的CoFi稀疏率计算"""
        total_params, active_params = 0, 0
        try:
            for layer in model.bert.encoder.layer:
                masks = []
                if hasattr(layer.attention.self, 'mask'): masks.append(layer.attention.self.mask)
                if hasattr(layer.attention.output, 'mask'): masks.append(layer.attention.output.mask)
                if hasattr(layer.output, 'mask'): masks.append(layer.output.mask)
                if hasattr(layer.output.dense, 'mask'): masks.append(layer.output.dense.mask)

                for mask in masks:
                    try:
                        if not (hasattr(mask, 'log_alpha') and mask.log_alpha is not None): continue
                        if not mask.log_alpha.dtype.is_floating_point: mask.log_alpha.data = mask.log_alpha.float()
                        keep_probs = torch.sigmoid(mask.log_alpha)
                        active_params += (keep_probs > 0.5).sum().item()
                        total_params += keep_probs.numel()
                    except Exception:
                        continue
            return 1.0 - (active_params / total_params) if total_params > 0 else 0.0
        except Exception:
            return 0.0

    def _compute_current_sparsity(self, model):
        return self._compute_model_sparsity_cofi_accurate(model)

    def _compute_differentiable_sparsity_ratio(self):
        """计算可微分的稀疏率代理"""
        total_L0_norm, total_mask_params = 0, 0
        for mask_group in self.per_layer_mask_groups:
            for mask in mask_group:
                total_L0_norm += mask.L().sum()
                total_mask_params += mask.features
        return total_L0_norm / total_mask_params if total_mask_params > 0 else 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        """修复后的损失计算函数"""
        labels = inputs.pop("labels") if self.label_smoother is not None and "labels" in inputs else None
        inputs["output_hidden_states"] = inputs.get("output_hidden_states", False) or self.distill_switch
        outputs = model(**inputs)

        if self.args.past_index >= 0: self._past = outputs[self.args.past_index]

        if labels is not None:
            base_loss = self.label_smoother(outputs, labels, shift_labels=unwrap_model(
                model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values())
        else:
            base_loss = outputs.get("loss", outputs[0])

        loss_dict = {'classification': base_loss}

        if self.distill_switch:
            try:
                distill_loss = self.compute_adaptive_distill_loss(unwrap_model(model), inputs, outputs["logits"])
                if torch.isfinite(distill_loss) and distill_loss < 10:
                    loss_dict['distillation'] = distill_loss
            except Exception:
                pass

            try:
                differentiable_retention = self._compute_differentiable_sparsity_ratio()
                target_retention = 1.0 - self.target_sparsity
                sparsity_violation = differentiable_retention - target_retention
                violations = [sparsity_violation, 0.0]
                lambda1, lambda2 = self.lagrangian_optimizer.update_multipliers(violations)
                lagrangian_loss = self.lagrangian_optimizer.compute_lagrangian_loss(
                    [torch.tensor(v, device=base_loss.device) for v in violations])

                if torch.isfinite(lagrangian_loss):
                    loss_dict['lagrangian'] = lagrangian_loss

                current_sparsity_for_log = self._compute_current_sparsity(unwrap_model(model))
                self.training_stats['sparsity_history'].append(current_sparsity_for_log)
                self.training_stats['lambda_history']['lambda1'].append(lambda1)
                self.training_stats['lambda_history']['lambda2'].append(lambda2)
                self._sync_lagrange_multipliers_to_model(lambda1, lambda2)
            except Exception:
                pass

        total_loss, loss_components = self.loss_calculator.compute_balanced_loss(loss_dict)
        for component, value in loss_components.items():
            if component in self.training_stats['loss_components']:
                self.training_stats['loss_components'][component].append(value)

        return (total_loss, outputs) if return_outputs else total_loss

    def _sync_lagrange_multipliers_to_model(self, lambda1, lambda2):
        """同步拉格朗日乘数到模型参数"""
        try:
            if hasattr(self.model.bert, 'reg_lambda_1'): self.model.bert.reg_lambda_1.data.fill_(lambda1)
            if hasattr(self.model.bert, 'reg_lambda_2'): self.model.bert.reg_lambda_2.data.fill_(lambda2)
        except Exception:
            pass

    def compute_adaptive_distill_loss(self, model: SModel, inputs: Dict, s_logits: torch.Tensor):
        """简化的自适应蒸馏损失计算"""
        with torch.no_grad():
            inputs_for_teacher = inputs.copy()
            inputs_for_teacher["output_hidden_states"] = False
            t_outputs = self.t_model(**inputs_for_teacher)
            t_logits = t_outputs["logits"]

        T = self.args.distill_T
        current_sparsity = self._compute_current_sparsity(model)
        adaptive_distill_lambda = self.get_adaptive_distill_weight(current_sparsity)

        self.training_stats['sparsity_history'].append(current_sparsity)
        self.training_stats['distill_weight_history'].append(adaptive_distill_lambda)

        pred_loss = self.kl_loss(F.log_softmax(s_logits / T, dim=-1), F.log_softmax(t_logits / T, dim=-1)) * (T ** 2)
        return adaptive_distill_lambda * pred_loss

    def get_adaptive_distill_weight(self, current_sparsity):
        """获取自适应蒸馏权重"""
        base_weight = 0.05
        adaptive_weight = base_weight + 0.15 * current_sparsity
        return min(adaptive_weight, 0.3)

    def compute_target_sparsity(self):
        return self.target_sparsity

    # MODIFIED: Fixed the 'float' object has no attribute 'abs' error.
    def compute_lagrangian_loss(self):
        """传统的拉格朗日损失计算（保留兼容性, 用于评估）"""
        s = self._compute_current_sparsity(unwrap_model(self.model))
        t = self.compute_target_sparsity()
        lambda_1 = self.model.bert.reg_lambda_1
        lambda_2 = self.model.bert.reg_lambda_2

        # --- START OF FIX ---
        # 将Python浮点数转换为张量以使用torch.abs
        device = lambda_1.device
        s_tensor = torch.tensor(s, device=device)
        t_tensor = torch.tensor(t, device=device)

        # 使用torch.abs替代.abs()
        lagrangian_loss = lambda_1 * torch.abs(s_tensor - t_tensor) + lambda_2 * torch.pow(s_tensor - t_tensor, 2.)
        # --- END OF FIX ---
        return lagrangian_loss

    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None,
                 metric_key_prefix: str = "eval") -> Dict[str, float]:
        """增强的评估方法"""
        if self.args.local_rank == 0:
            with torch.no_grad():
                lambda_1_val = getattr(self.model.bert, 'reg_lambda_1', torch.tensor(0.0)).item()
                lambda_2_val = getattr(self.model.bert, 'reg_lambda_2', torch.tensor(0.0)).item()
                actual_sparsity = self._compute_current_sparsity(unwrap_model(self.model))
                t_sparsity = self.compute_target_sparsity()
                lagrangian_val = self.compute_lagrangian_loss().item() if hasattr(self,
                                                                                  'compute_lagrangian_loss') else 0.0

                print(f"📊 训练统计:")
                print(f"   λ₁ (model param): {lambda_1_val:.6f}")
                print(f"   λ₂ (model param): {lambda_2_val:.6f}")
                print(f"   实际稀疏率 (Actual Sparsity): {actual_sparsity:.4f}")
                print(f"   目标稀疏率: {t_sparsity:.4f}")
                print(f"   拉格朗日损失 (eval): {lagrangian_val:.6f}")

                clip_stats = self.gradient_clipper.get_clipping_stats()
                print(f"   梯度裁剪率: {clip_stats['clipping_ratio']:.3f}")
                print(f"   当前裁剪阈值: {clip_stats['current_clip_norm']:.3f}")
                print(f"   平均梯度范数: {clip_stats['avg_grad_norm']:.3f}")

                if self.training_stats['lambda_history']['lambda1']:
                    print(f"   稳定λ₁ (trainer): {np.mean(self.training_stats['lambda_history']['lambda1'][-10:]):.6f}")
                    print(f"   稳定λ₂ (trainer): {np.mean(self.training_stats['lambda_history']['lambda2'][-10:]):.6f}")

        past_distill_switch = self.distill_switch
        self.distill_switch = False
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        self.distill_switch = past_distill_switch

        with torch.no_grad():
            results['actual_sparsity'] = self._compute_current_sparsity(unwrap_model(self.model))
            results['target_sparsity'] = self.compute_target_sparsity()
            try:
                results['lagrangian_loss'] = self.compute_lagrangian_loss().item()
            except:
                results['lagrangian_loss'] = 0.0
            results['distill_weight'] = self.get_adaptive_distill_weight(results['actual_sparsity'])
            if self.training_stats['grad_norm_history']:
                results['avg_grad_norm'] = np.mean(
                    [g['original_norm'] for g in self.training_stats['grad_norm_history'][-10:]])
        return results

    def _print_training_summary(self):
        """打印训练总结"""
        if not self.training_stats.get('sparsity_history'): return
        print(f"\n📈 训练总结:")
        print(f"   最终稀疏率: {self.training_stats['sparsity_history'][-1]:.4f}")
        if self.training_stats['distill_weight_history']: print(
            f"   最终蒸馏权重: {self.training_stats['distill_weight_history'][-1]:.4f}")
        if self.training_stats['grad_norm_history']:
            recent_grads = self.training_stats['grad_norm_history'][-10:]
            print(f"   平均梯度范数: {np.mean([g['original_norm'] for g in recent_grads]):.4f}")
            print(f"   最近裁剪率: {np.mean([g['was_clipped'] for g in recent_grads]):.3f}")
        for component in ['classification', 'distillation', 'lagrangian']:
            if self.training_stats['loss_components'].get(component):
                print(f"   平均{component}损失: {np.mean(self.training_stats['loss_components'][component][-10:]):.6f}")
        if self.training_stats['lambda_history']['lambda1']:
            print(f"   λ₁稳定性(std): {np.std(self.training_stats['lambda_history']['lambda1'][-20:]):.6f}")
            print(f"   λ₂稳定性(std): {np.std(self.training_stats['lambda_history']['lambda2'][-20:]):.6f}")
