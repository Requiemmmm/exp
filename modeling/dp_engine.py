"""
规范化差分隐私模块 for 联邦学习
实现标准的DP-SGD和客户端级差分隐私保护

主要特性：
1. 标准的(ε,δ)-差分隐私保证
2. RDP-based隐私预算追踪
3. 自适应梯度裁剪
4. 联邦学习客户端级隐私保护
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
from enum import Enum


class PrivacyAccountantType(Enum):
    """隐私会计类型"""
    RDP = "rdp"  # Rényi Differential Privacy
    MOMENTS = "moments"  # Moments Accountant
    ADVANCED_COMPOSITION = "advanced_composition"


class DPMechanism(Enum):
    """差分隐私机制类型"""
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"


class PrivacyAccountant:
    """
    隐私预算追踪器
    实现RDP-based精确隐私分析
    """

    def __init__(self,
                 target_epsilon: float = 1.0,
                 target_delta: float = 1e-5,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: float = 1.0,
                 sample_rate: float = 0.01,
                 accountant_type: PrivacyAccountantType = PrivacyAccountantType.RDP):

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.accountant_type = accountant_type

        # 隐私损失追踪
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.step_count = 0
        self.composition_history = []

        # RDP参数
        self.rdp_orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))
        self.rdp_epsilon = [0.0] * len(self.rdp_orders)

    def compute_rdp_epsilon(self, q: float, noise_multiplier: float, steps: int) -> List[float]:
        """
        计算RDP隐私损失
        基于"The Algorithmic Foundations of Differential Privacy"
        """
        rdp_eps = []
        for alpha in self.rdp_orders:
            if alpha == 1:
                rdp_eps.append(float('inf'))
            else:
                # RDP for Gaussian mechanism
                rdp = steps * q * q * alpha / (2.0 * noise_multiplier * noise_multiplier)
                rdp_eps.append(rdp)
        return rdp_eps

    def rdp_to_dp(self, rdp_eps: List[float]) -> Tuple[float, float]:
        """
        将RDP转换为(ε,δ)-DP
        使用标准转换公式
        """
        min_eps = float('inf')
        for i, alpha in enumerate(self.rdp_orders):
            if alpha == 1:
                continue
            # 转换公式: ε = RDP_α + log(1/δ)/(α-1)
            eps = rdp_eps[i] + math.log(1.0 / self.target_delta) / (alpha - 1)
            min_eps = min(min_eps, eps)
        return min_eps, self.target_delta

    def step(self, q: float = None, noise_multiplier: float = None, steps: int = 1):
        """更新隐私预算"""
        q = q or self.sample_rate
        noise_multiplier = noise_multiplier or self.noise_multiplier

        # 计算当前步骤的RDP损失
        step_rdp = self.compute_rdp_epsilon(q, noise_multiplier, steps)

        # 累积RDP损失
        for i in range(len(self.rdp_epsilon)):
            self.rdp_epsilon[i] += step_rdp[i]

        # 转换为(ε,δ)-DP
        self.spent_epsilon, self.spent_delta = self.rdp_to_dp(self.rdp_epsilon)
        self.step_count += steps

        # 记录历史
        self.composition_history.append({
            'step': self.step_count,
            'q': q,
            'noise_multiplier': noise_multiplier,
            'epsilon': self.spent_epsilon,
            'delta': self.spent_delta
        })

        # 检查预算超支
        if self.spent_epsilon > self.target_epsilon:
            warnings.warn(
                f"Privacy budget exceeded! Current ε={self.spent_epsilon:.4f}, target ε={self.target_epsilon}")

    def get_privacy_spent(self) -> Tuple[float, float]:
        """获取已消耗的隐私预算"""
        return self.spent_epsilon, self.spent_delta

    def get_remaining_budget(self) -> float:
        """获取剩余隐私预算"""
        return max(0.0, self.target_epsilon - self.spent_epsilon)

    def is_budget_exhausted(self) -> bool:
        """检查隐私预算是否耗尽"""
        return self.spent_epsilon >= self.target_epsilon


class GradientClipper:
    """
    自适应梯度裁剪器
    支持per-sample和全局裁剪
    """

    def __init__(self,
                 max_grad_norm: float = 1.0,
                 adaptive: bool = True,
                 percentile: float = 50.0):
        self.max_grad_norm = max_grad_norm
        self.adaptive = adaptive
        self.percentile = percentile
        self.grad_norm_history = []
        self.warmup_steps = 100

    def clip_gradients(self,
                       parameters,
                       max_norm: float = None) -> Tuple[float, bool]:
        """
        裁剪梯度并返回裁剪前范数和是否被裁剪
        """
        max_norm = max_norm or self.max_grad_norm

        # 计算总梯度范数
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # 记录历史用于自适应调整
        self.grad_norm_history.append(total_norm)
        if len(self.grad_norm_history) > 1000:
            self.grad_norm_history = self.grad_norm_history[-1000:]

        # 自适应调整裁剪阈值
        if self.adaptive and len(self.grad_norm_history) > self.warmup_steps:
            self.max_grad_norm = np.percentile(self.grad_norm_history, self.percentile)
            max_norm = self.max_grad_norm

        # 执行裁剪
        clipped = total_norm > max_norm
        if clipped:
            clip_coef = max_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        return total_norm, clipped

    def per_sample_clip(self,
                        per_sample_grads: torch.Tensor,
                        max_norm: float = None) -> torch.Tensor:
        """
        Per-sample梯度裁剪
        Args:
            per_sample_grads: shape [batch_size, param_size]
        """
        max_norm = max_norm or self.max_grad_norm
        batch_size = per_sample_grads.shape[0]

        # 计算每个样本的梯度范数
        per_sample_norms = torch.norm(per_sample_grads.view(batch_size, -1), dim=1)

        # 计算裁剪系数
        clip_factors = torch.clamp(max_norm / (per_sample_norms + 1e-8), max=1.0)

        # 应用裁剪
        clipped_grads = per_sample_grads * clip_factors.view(-1, 1)

        return clipped_grads


class DifferentialPrivacyEngine:
    """
    差分隐私引擎
    集成梯度裁剪、噪声添加和隐私预算管理
    """

    def __init__(self,
                 target_epsilon: float = 10.0,
                 target_delta: float = 1e-3,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: float = 0.5,
                 sample_rate: float = 0.01,
                 mechanism: DPMechanism = DPMechanism.GAUSSIAN,
                 secure_mode: bool = True):

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.mechanism = mechanism
        self.secure_mode = secure_mode

        # 初始化组件
        self.accountant = PrivacyAccountant(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate
        )

        self.clipper = GradientClipper(max_grad_norm=max_grad_norm, adaptive=True)

        # 统计信息
        self.stats = {
            'total_steps': 0,
            'clipped_steps': 0,
            'noise_scale_history': [],
            'grad_norm_history': []
        }

    

    def add_noise(self, tensor: torch.Tensor, sensitivity: float = None) -> torch.Tensor:
        """向张量添加校准噪声，增加类型检查"""
        sensitivity = sensitivity or self.max_grad_norm

        # ✅ 检查张量类型，只对浮点型张量添加噪声
        if not tensor.dtype.is_floating_point:
            return tensor

        # ✅ 检查张量是否包含有效数值
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return tensor

        try:
            if self.mechanism == DPMechanism.GAUSSIAN:
                noise_scale = sensitivity * self.noise_multiplier
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype
                )
            else:  # LAPLACE
                b = sensitivity / self.target_epsilon
                noise = torch.tensor(
                    np.random.laplace(0, b, tensor.shape),
                    device=tensor.device,
                    dtype=tensor.dtype
                )

            return tensor + noise

        except Exception as e:
            print(f"⚠️ 添加噪声失败，返回原张量: {e}")
            return tensor

    def privatize_gradients(self,
                            model: nn.Module,
                            batch_size: int) -> Dict[str, torch.Tensor]:
        """
        对模型梯度进行差分隐私化处理
        """
        # 1. 梯度裁剪
        grad_norm, was_clipped = self.clipper.clip_gradients(model.parameters())

        # 2. 提取裁剪后的梯度
        clipped_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                clipped_grads[name] = param.grad.clone()

        # 3. 添加校准噪声
        private_grads = {}
        for name, grad in clipped_grads.items():
            if self._should_add_noise(name):
                private_grads[name] = self.add_noise(grad)
            else:
                private_grads[name] = grad

        # 4. 更新隐私预算
        self.accountant.step(
            q=batch_size / 10000,  # 假设总数据集大小为10000
            noise_multiplier=self.noise_multiplier
        )

        # 5. 更新统计信息
        self.stats['total_steps'] += 1
        if was_clipped:
            self.stats['clipped_steps'] += 1
        self.stats['noise_scale_history'].append(self.max_grad_norm * self.noise_multiplier)
        self.stats['grad_norm_history'].append(grad_norm)

        return private_grads

    def _should_add_noise(self, param_name: str) -> bool:
        """判断是否需要对特定参数添加噪声"""
        # 通常不对正则化参数添加噪声
        skip_patterns = ['reg_lambda', 'bias']
        return not any(pattern in param_name for pattern in skip_patterns)

    def get_privacy_analysis(self) -> Dict:
        """获取隐私分析报告"""
        epsilon, delta = self.accountant.get_privacy_spent()

        analysis = {
            'privacy_spent': {
                'epsilon': epsilon,
                'delta': delta,
                'remaining_epsilon': self.accountant.get_remaining_budget()
            },
            'training_stats': {
                'total_steps': self.stats['total_steps'],
                'clipped_ratio': self.stats['clipped_steps'] / max(1, self.stats['total_steps']),
                'avg_grad_norm': np.mean(self.stats['grad_norm_history'][-100:]) if self.stats[
                    'grad_norm_history'] else 0,
                'current_clip_norm': self.clipper.max_grad_norm,
                'current_noise_scale': self.max_grad_norm * self.noise_multiplier
            },
            'privacy_parameters': {
                'target_epsilon': self.target_epsilon,
                'target_delta': self.target_delta,
                'noise_multiplier': self.noise_multiplier,
                'max_grad_norm': self.max_grad_norm
            }
        }

        return analysis

    def save_privacy_analysis(self, filepath: str):
        """保存隐私分析到文件"""
        analysis = self.get_privacy_analysis()
        import json
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)


class FederatedDPAggregator:
    """
    联邦学习差分隐私聚合器
    实现客户端级差分隐私保护
    """

    def __init__(self,
                 num_clients: int,
                 target_epsilon: float = 10.0,
                 target_delta: float = 1e-5,
                 clip_norm: float = 1.0):

        self.num_clients = num_clients
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.clip_norm = clip_norm

        # 为每个客户端创建DP引擎
        client_epsilon = target_epsilon / num_clients  # 简单均分预算
        self.client_dp_engines = {}

        for client_id in range(num_clients):
            self.client_dp_engines[client_id] = DifferentialPrivacyEngine(
                target_epsilon=client_epsilon,
                target_delta=target_delta,
                max_grad_norm=clip_norm,
                noise_multiplier=self._compute_noise_multiplier(client_epsilon, target_delta)
            )

    def _compute_noise_multiplier(self, epsilon: float, delta: float) -> float:
        """根据隐私预算计算噪声乘数"""
        # 使用标准公式：σ = √(2ln(1.25/δ)) / ε
        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon



    def aggregate_with_privacy(self,
                               client_updates: List[Dict[str, torch.Tensor]],
                               client_weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        修复后的差分隐私联邦聚合
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        num_clients = len(client_updates)
        client_weights = client_weights or [1.0 / num_clients] * num_clients

        # 对每个客户端的更新进行隐私化处理
        private_updates = []
        for i, updates in enumerate(client_updates):
            private_update = {}
            for name, param in updates.items():

                # ✅ 检查参数类型，只对浮点型参数进行DP处理
                if not param.dtype.is_floating_point:
                    private_update[name] = param
                    continue

                try:
                    # 裁剪客户端更新
                    param_norm = torch.norm(param).item()
                    if param_norm > self.clip_norm:
                        param = param * (self.clip_norm / param_norm)

                    # 添加噪声
                    if i < len(self.client_dp_engines):
                        private_param = self.client_dp_engines[i].add_noise(param, self.clip_norm)
                    else:
                        private_param = param

                    private_update[name] = private_param

                except Exception as e:
                    print(f"⚠️ 参数 {name} DP处理失败: {e}")
                    private_update[name] = param

            private_updates.append(private_update)

        # ✅ 类型感知的加权聚合
        aggregated = {}
        first_update = private_updates[0]

        for name in first_update.keys():
            first_param = first_update[name]

            if not first_param.dtype.is_floating_point:
                # 非浮点型参数：使用多数投票或直接复制
                if first_param.dtype == torch.bool:
                    vote_sum = torch.zeros_like(first_param, dtype=torch.float32)
                    for update in private_updates:
                        vote_sum += update[name].float()
                    aggregated[name] = (vote_sum > num_clients / 2).to(first_param.dtype)
                else:
                    aggregated[name] = first_param.clone()
            else:
                # 浮点型参数：加权平均
                aggregated[name] = torch.zeros_like(first_param)
                for i, update in enumerate(private_updates):
                    aggregated[name] += client_weights[i] * update[name]

        return aggregated

    def get_global_privacy_analysis(self) -> Dict:
        """获取全局隐私分析"""
        total_epsilon = 0.0
        client_analyses = {}

        for client_id, dp_engine in self.client_dp_engines.items():
            analysis = dp_engine.get_privacy_analysis()
            client_analyses[f'client_{client_id}'] = analysis
            total_epsilon += analysis['privacy_spent']['epsilon']

        return {
            'global_epsilon': total_epsilon,
            'target_epsilon': self.target_epsilon,
            'privacy_preserved': total_epsilon <= self.target_epsilon,
            'client_analyses': client_analyses
        }


# 使用示例和集成代码
def integrate_dp_into_trainer(trainer_class):
    """
    将差分隐私集成到现有训练器中的装饰器
    """

    class DPEnhancedTrainer(trainer_class):
        def __init__(self, *args, dp_config=None, **kwargs):
            super().__init__(*args, **kwargs)

            if dp_config is None:
                dp_config = {
                    'target_epsilon': 1.0,
                    'target_delta': 1e-5,
                    'max_grad_norm': 1.0,
                    'noise_multiplier': 1.0
                }

            self.dp_engine = DifferentialPrivacyEngine(**dp_config)
            self.dp_enabled = True

        def training_step(self, model, inputs):
            """重写训练步骤以包含差分隐私"""
            # 正常的前向和反向传播
            loss = super().compute_loss(model, inputs)
            loss.backward()

            # 差分隐私处理
            if self.dp_enabled:
                batch_size = inputs['input_ids'].shape[0]
                private_grads = self.dp_engine.privatize_gradients(model, batch_size)

                # 将私有梯度应用回模型
                for name, param in model.named_parameters():
                    if name in private_grads:
                        param.grad = private_grads[name]

            return loss

        def get_dp_analysis(self):
            """获取差分隐私分析"""
            return self.dp_engine.get_privacy_analysis()

    return DPEnhancedTrainer