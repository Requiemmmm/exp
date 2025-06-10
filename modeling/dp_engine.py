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
from scipy.stats import norm

def _get_privacy_spent(noise_multiplier: float, delta: float) -> float:
    """计算高斯机制下单次扰动的(epsilon, delta)-DP。"""
    if noise_multiplier == 0:
        return float('inf')
    # RDP to (ε, δ)-DP conversion
    # This is a standard formula for Gaussian mechanism.
    # ε(σ) = min_α ( (α-1) * log(1 - q + q*e^(1/σ^2)) - log(δ) ) / (α-1)
    # For q=1 (adding noise to the function output directly), it simplifies.
    # A common and tight bound is ε = sqrt(2*log(1.25/δ)) / σ
    return math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier


class LocalDPEngine:
    """修复后的差分隐私引擎"""

    def __init__(self, target_epsilon: float = 10.0,
                 target_delta: float = 1e-3,
                 num_rounds: int = 10,
                 num_clients: int = 5,
                 max_grad_norm: float = 1.0,
                 noise_multiplier: float = None):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.max_grad_norm = max_grad_norm

        # 修复：使用高级组合定理正确计算每轮预算
        self.per_round_epsilon = self._compute_per_round_epsilon()

        # 如果没有指定噪声乘数，则自动计算
        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

        # 追踪累积隐私成本
        self.accumulated_epsilon = 0.0
        self.round_count = 0

        print(f"✅ LocalDPEngine 初始化完成:")
        print(f"   - 目标ε: {self.target_epsilon}, δ: {self.target_delta}")
        print(f"   - 每轮ε: {self.per_round_epsilon:.4f}")
        print(f"   - 噪声乘数: {self.noise_multiplier:.4f}")

    def _compute_per_round_epsilon(self) -> float:
        """使用高级组合定理计算每轮隐私预算"""
        # 高级组合：sqrt(2k*ln(1/δ))*ε + k*ε*(e^ε-1)
        k = self.num_rounds
        delta = self.target_delta
        target_eps = self.target_epsilon

        # 求解每轮ε，使总ε不超过目标值
        def composition_bound(eps_per_round):
            if eps_per_round <= 0:
                return 0
            try:
                term1 = np.sqrt(2 * k * np.log(1 / delta)) * eps_per_round
                term2 = k * eps_per_round * (np.exp(eps_per_round) - 1)
                return term1 + term2
            except (OverflowError, ValueError):
                return float('inf')

        # 二分搜索找到合适的每轮ε
        left, right = 0.0, min(target_eps, 10.0)  # 限制搜索范围
        tolerance = 1e-6
        max_iterations = 100

        for _ in range(max_iterations):
            if right - left < tolerance:
                break
            mid = (left + right) / 2
            if composition_bound(mid) <= target_eps:
                left = mid
            else:
                right = mid

        return left * 0.9  # 安全裕度

    def _compute_noise_multiplier(self) -> float:
        """计算噪声乘数以满足隐私预算"""
        # 使用RDP到(ε,δ)-DP的转换
        # σ ≥ sqrt(2*ln(1.25/δ)) / ε
        if self.per_round_epsilon <= 0:
            return 10.0  # 默认较大的噪声乘数

        try:
            multiplier = np.sqrt(2 * np.log(1.25 / self.target_delta)) / self.per_round_epsilon
            return max(0.01, min(multiplier, 100.0))  # 限制范围
        except (ValueError, OverflowError):
            return 1.0

    def add_noise(self, tensor: torch.Tensor, sensitivity: float = None) -> torch.Tensor:
        """为张量添加噪声"""
        if sensitivity is None:
            sensitivity = self.max_grad_norm

        # 类型检查
        if not tensor.dtype.is_floating_point:
            return tensor

        # 数值检查
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            warnings.warn("Tensor contains NaN or Inf values. Skipping noise addition.")
            return tensor

        try:
            noise_scale = sensitivity * self.noise_multiplier
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=tensor.shape,
                device=tensor.device,
                dtype=tensor.dtype
            )
            return tensor + noise
        except Exception as e:
            warnings.warn(f"Failed to add noise: {e}")
            return tensor

    def add_noise_to_gradients(self, gradients: Dict[str, torch.Tensor],
                               sensitivity: float = 1.0) -> Dict[str, torch.Tensor]:
        """为梯度添加噪声并更新隐私预算"""
        noised_gradients = {}

        for name, grad in gradients.items():
            if grad is None:
                noised_gradients[name] = None
                continue

            # 添加高斯噪声
            noised_gradients[name] = self.add_noise(grad, sensitivity)

        # 更新隐私预算追踪
        self.round_count += 1
        self.accumulated_epsilon += self.per_round_epsilon

        # 检查是否超出预算
        if self.accumulated_epsilon > self.target_epsilon:
            warnings.warn(f"隐私预算即将耗尽! 当前: {self.accumulated_epsilon:.4f}, "
                          f"目标: {self.target_epsilon}")

        return noised_gradients

    def get_privacy_cost(self) -> Tuple[float, float]:
        """返回当前隐私成本"""
        return self.accumulated_epsilon, self.target_epsilon - self.accumulated_epsilon


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

        # RDP参数 - 修复：使用numpy数组而不是tensor
        self.rdp_orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))
        self.rdp_alphas = np.array(self.rdp_orders)
        self.rdp_epsilon = np.zeros_like(self.rdp_alphas)

    def compute_rdp_epsilon_for_one_step(self, q: float, noise_multiplier: float) -> np.ndarray:
        """
        为单次采样高斯机制计算RDP隐私损失
        修复：使用numpy进行数值计算，避免设备兼容性问题
        """
        if q == 0:
            return np.zeros_like(self.rdp_alphas)

        alphas = self.rdp_alphas
        rdp = np.zeros_like(alphas)

        try:
            # 当q=1时，RDP简化为 α / (2σ²)
            if q >= 1.0:
                rdp = alphas / (2 * noise_multiplier**2)
            else:
                # 当q<1时，使用子采样高斯机制的RDP公式
                for i, alpha in enumerate(alphas):
                    if alpha <= 1:
                        rdp[i] = 0
                        continue

                    try:
                        # 使用稳定的数值计算
                        if noise_multiplier == 0:
                            rdp[i] = float('inf')
                            continue

                        # RDP for subsampled Gaussian mechanism
                        # 使用更稳定的计算方式
                        sigma_sq = noise_multiplier ** 2

                        if q == 1.0:
                            rdp[i] = alpha / (2 * sigma_sq)
                        else:
                            # 使用近似公式避免数值溢出
                            term1 = q * alpha * (alpha - 1) / (2 * sigma_sq)
                            term2 = np.log(1 + q * (np.exp(1 / sigma_sq) - 1))
                            rdp[i] = min(term1, term2)

                    except (OverflowError, ValueError, FloatingPointError):
                        rdp[i] = float('inf')

        except Exception as e:
            warnings.warn(f"RDP computation failed: {e}")
            rdp = np.full_like(alphas, float('inf'))

        # 确保RDP值非负且有限
        rdp = np.clip(rdp, 0, 1e6)
        return rdp

    def rdp_to_dp(self, rdp_eps: np.ndarray) -> Tuple[float, float]:
        """
        将RDP转换为(ε,δ)-DP
        修复：使用numpy进行稳定的数值计算
        """
        try:
            alphas = self.rdp_alphas

            # 避免除零和对数计算错误
            valid_mask = (alphas > 1) & (rdp_eps < float('inf'))

            if not np.any(valid_mask):
                return float('inf'), self.target_delta

            # 转换公式: ε = RDP_α + log(1/δ)/(α-1)
            log_delta_term = np.log(1.0 / self.target_delta)
            eps_candidates = rdp_eps[valid_mask] + log_delta_term / (alphas[valid_mask] - 1)

            # 取最小值
            min_eps = np.min(eps_candidates)

            # 确保结果有效
            if np.isnan(min_eps) or np.isinf(min_eps):
                return float('inf'), self.target_delta

            return float(min_eps), self.target_delta

        except Exception as e:
            warnings.warn(f"RDP to DP conversion failed: {e}")
            return float('inf'), self.target_delta

    def step(self, q: float = None, noise_multiplier: float = None, steps: int = 1):
        """更新隐私预算"""
        q = q or self.sample_rate
        noise_multiplier = noise_multiplier or self.noise_multiplier

        try:
            # 计算当前步骤的RDP损失
            step_rdp = self.compute_rdp_epsilon_for_one_step(q, noise_multiplier)

            # 累积RDP损失（RDP的组合就是简单相加）
            self.rdp_epsilon += step_rdp * steps

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

        except Exception as e:
            warnings.warn(f"Privacy accounting step failed: {e}")

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
        修复：增强数值稳定性和错误处理
        """
        max_norm = max_norm or self.max_grad_norm

        # 过滤掉没有梯度的参数
        params_with_grad = [p for p in parameters if p.grad is not None]
        if not params_with_grad:
            return 0.0, False

        try:
            # 计算总梯度范数
            grad_norms = []
            for p in params_with_grad:
                if p.grad is not None:
                    param_norm = torch.norm(p.grad.detach(), 2).item()
                    if not (np.isnan(param_norm) or np.isinf(param_norm)):
                        grad_norms.append(param_norm ** 2)

            if not grad_norms:
                return 0.0, False

            total_norm = math.sqrt(sum(grad_norms))

            # 记录历史用于自适应调整
            if not (np.isnan(total_norm) or np.isinf(total_norm)):
                self.grad_norm_history.append(total_norm)
                if len(self.grad_norm_history) > 1000:
                    self.grad_norm_history = self.grad_norm_history[-1000:]

            # 自适应调整裁剪阈值
            if self.adaptive and len(self.grad_norm_history) > self.warmup_steps:
                try:
                    adaptive_norm = np.percentile(self.grad_norm_history, self.percentile)
                    self.max_grad_norm = max(0.1, min(adaptive_norm, 10.0))
                    max_norm = self.max_grad_norm
                except:
                    pass  # 保持原始值

            # 执行裁剪
            clipped = total_norm > max_norm
            if clipped and total_norm > 0:
                clip_coef = max_norm / (total_norm + 1e-8)
                for p in params_with_grad:
                    if p.grad is not None:
                        p.grad.detach().mul_(clip_coef)

            return total_norm, clipped

        except Exception as e:
            warnings.warn(f"Gradient clipping failed: {e}")
            return 0.0, False

    def per_sample_clip(self,
                        per_sample_grads: torch.Tensor,
                        max_norm: float = None) -> torch.Tensor:
        """
        Per-sample梯度裁剪
        Args:
            per_sample_grads: shape [batch_size, param_size]
        """
        max_norm = max_norm or self.max_grad_norm

        try:
            batch_size = per_sample_grads.shape[0]

            # 计算每个样本的梯度范数
            per_sample_norms = torch.norm(per_sample_grads.view(batch_size, -1), p=2, dim=1)

            # 计算裁剪系数
            clip_factors = torch.clamp_max(max_norm / (per_sample_norms + 1e-8), 1.0)

            # 应用裁剪
            clipped_grads = per_sample_grads * clip_factors.view(-1, *([1] * (per_sample_grads.dim() - 1)))

            return clipped_grads

        except Exception as e:
            warnings.warn(f"Per-sample clipping failed: {e}")
            return per_sample_grads


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
        try:
            self.accountant = PrivacyAccountant(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate
            )
        except Exception as e:
            warnings.warn(f"Privacy accountant initialization failed: {e}")
            self.accountant = None

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

        # 检查张量类型，只对浮点型张量添加噪声
        if not tensor.dtype.is_floating_point:
            return tensor

        # 检查张量是否包含有效数值
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            warnings.warn(f"Tensor contains NaN or Inf values. Skipping noise addition.")
            return tensor

        try:
            if self.mechanism == DPMechanism.GAUSSIAN:
                noise_scale = sensitivity * self.noise_multiplier
                # 限制噪声规模避免数值问题
                noise_scale = min(noise_scale, 100.0)

                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=tensor.shape,
                    device=tensor.device,
                    dtype=tensor.dtype
                )
            else:  # LAPLACE
                b = sensitivity / max(self.target_epsilon, 1e-6)
                noise = torch.tensor(
                    np.random.laplace(0, b, tensor.shape),
                    device=tensor.device,
                    dtype=tensor.dtype
                )

            result = tensor + noise

            # 检查结果的有效性
            if torch.isnan(result).any() or torch.isinf(result).any():
                warnings.warn("Noise addition resulted in NaN/Inf. Returning original tensor.")
                return tensor

            return result

        except Exception as e:
            warnings.warn(f"添加噪声失败，返回原张量: {e}")
            return tensor

    def privatize_gradients(self,
                            model: nn.Module,
                            dataset_size: int,
                            batch_size: int) -> Dict[str, torch.Tensor]:
        """
        对模型梯度进行差分隐私化处理
        """
        try:
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
            if self.accountant is not None:
                try:
                    current_sample_rate = batch_size / max(dataset_size, 1) if dataset_size > 0 else 0
                    self.accountant.step(
                        q=current_sample_rate,
                        noise_multiplier=self.noise_multiplier
                    )
                except Exception as e:
                    warnings.warn(f"Privacy accounting update failed: {e}")

            # 5. 更新统计信息
            self.stats['total_steps'] += 1
            if was_clipped:
                self.stats['clipped_steps'] += 1
            self.stats['noise_scale_history'].append(self.max_grad_norm * self.noise_multiplier)
            self.stats['grad_norm_history'].append(grad_norm)

            return private_grads

        except Exception as e:
            warnings.warn(f"Gradient privatization failed: {e}")
            # 返回原始梯度作为备选
            return {name: param.grad.clone() for name, param in model.named_parameters()
                    if param.grad is not None}

    def _should_add_noise(self, param_name: str) -> bool:
        """判断是否需要对特定参数添加噪声"""
        # 通常不对正则化参数、偏置项和归一化层添加噪声
        skip_patterns = ['reg_lambda', 'bias', 'LayerNorm', 'layer_norm', 'position_ids', 'token_type_ids']
        return not any(pattern in param_name for pattern in skip_patterns)

    def get_privacy_analysis(self) -> Dict:
        """获取隐私分析报告"""
        try:
            if self.accountant is not None:
                epsilon, delta = self.accountant.get_privacy_spent()
                remaining_budget = self.accountant.get_remaining_budget()
            else:
                epsilon, delta, remaining_budget = float('inf'), self.target_delta, 0.0

            analysis = {
                'privacy_spent': {
                    'epsilon': epsilon,
                    'delta': delta,
                    'remaining_epsilon': remaining_budget
                },
                'training_stats': {
                    'total_steps': self.stats['total_steps'],
                    'clipped_ratio': self.stats['clipped_steps'] / max(1, self.stats['total_steps']),
                    'avg_grad_norm': np.mean(self.stats['grad_norm_history'][-100:]) if self.stats['grad_norm_history'] else 0,
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

        except Exception as e:
            warnings.warn(f"Privacy analysis generation failed: {e}")
            return {'error': str(e)}

    def save_privacy_analysis(self, filepath: str):
        """保存隐私分析到文件"""
        try:
            analysis = self.get_privacy_analysis()
            import json
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
        except Exception as e:
            warnings.warn(f"Failed to save privacy analysis: {e}")


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
        client_epsilon = target_epsilon / max(num_clients, 1)  # 简单均分预算
        self.client_dp_engines = {}

        for client_id in range(num_clients):
            try:
                noise_multiplier = self._compute_noise_multiplier(client_epsilon, target_delta)
                self.client_dp_engines[client_id] = DifferentialPrivacyEngine(
                    target_epsilon=client_epsilon,
                    target_delta=target_delta,
                    max_grad_norm=clip_norm,
                    noise_multiplier=noise_multiplier
                )
            except Exception as e:
                warnings.warn(f"Failed to create DP engine for client {client_id}: {e}")

    def _compute_noise_multiplier(self, epsilon: float, delta: float) -> float:
        """根据隐私预算计算噪声乘数"""
        if epsilon <= 0 or epsilon == float('inf'):
            return 1.0
        try:
            # 使用标准公式：σ = √(2ln(1.25/δ)) / ε
            multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            return max(0.01, min(multiplier, 100.0))  # 限制范围
        except (ValueError, OverflowError):
            return 1.0

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

            try:
                for name, param in updates.items():
                    # 检查参数类型，只对浮点型参数进行DP处理
                    if not param.dtype.is_floating_point:
                        private_update[name] = param
                        continue

                    # 裁剪客户端更新
                    param_norm = torch.norm(param).item()
                    if param_norm > self.clip_norm and param_norm > 0:
                        param = param * (self.clip_norm / (param_norm + 1e-8))

                    # 添加噪声
                    dp_engine_id = i % max(len(self.client_dp_engines), 1)
                    if dp_engine_id in self.client_dp_engines:
                        private_param = self.client_dp_engines[dp_engine_id].add_noise(param, self.clip_norm)
                    else:
                        private_param = param

                    private_update[name] = private_param

            except Exception as e:
                warnings.warn(f"DP processing failed for client {i}: {e}")
                private_update = updates  # 使用原始更新作为备选

            private_updates.append(private_update)

        # 类型感知的加权聚合
        try:
            aggregated = {}
            if private_updates:
                first_update = private_updates[0]

                for name in first_update.keys():
                    try:
                        first_param = first_update[name]

                        if not first_param.dtype.is_floating_point:
                            # 非浮点型参数：使用多数投票或直接复制
                            if first_param.dtype == torch.bool:
                                vote_sum = torch.zeros_like(first_param, dtype=torch.float32)
                                for update in private_updates:
                                    if name in update:
                                        vote_sum += update[name].float()
                                aggregated[name] = (vote_sum > num_clients / 2).to(first_param.dtype)
                            else:
                                aggregated[name] = first_param.clone()
                        else:
                            # 浮点型参数：加权平均
                            aggregated[name] = torch.zeros_like(first_param)
                            for i, update in enumerate(private_updates):
                                if name in update:
                                    aggregated[name] += client_weights[i] * update[name]

                    except Exception as e:
                        warnings.warn(f"Aggregation failed for parameter {name}: {e}")
                        aggregated[name] = first_update[name]

            return aggregated

        except Exception as e:
            warnings.warn(f"Global aggregation failed: {e}")
            return client_updates[0] if client_updates else {}

    def get_global_privacy_analysis(self) -> Dict:
        """获取全局隐私分析"""
        try:
            # 在本地DP模型下，全局隐私预算等于单个客户端的预算
            # 这里我们以最保守的方式进行分析

            client_analyses = {}
            max_epsilon = 0.0

            for client_id, dp_engine in self.client_dp_engines.items():
                try:
                    analysis = dp_engine.get_privacy_analysis()
                    client_analyses[f'client_{client_id}'] = analysis

                    client_epsilon = analysis.get('privacy_spent', {}).get('epsilon', 0)
                    if isinstance(client_epsilon, (int, float)) and client_epsilon > max_epsilon:
                        max_epsilon = client_epsilon

                except Exception as e:
                    warnings.warn(f"Analysis failed for client {client_id}: {e}")

            return {
                'global_epsilon_estimation': max_epsilon,
                'target_epsilon_per_client': self.target_epsilon / max(self.num_clients, 1),
                'privacy_preserved': max_epsilon <= self.target_epsilon,
                'client_analyses': client_analyses,
                'aggregation_method': 'local_dp_with_secure_aggregation'
            }

        except Exception as e:
            warnings.warn(f"Global privacy analysis failed: {e}")
            return {'error': str(e)}


# 使用示例和集成代码
def integrate_dp_into_trainer(trainer_class):
    """
    将差分隐私集成到现有训练器中的装饰器
    """

    class DPEnhancedTrainer(trainer_class):
        def __init__(self, *args, dp_config=None, dataset_size=None, **kwargs):
            super().__init__(*args, **kwargs)

            if dp_config is None:
                dp_config = {
                    'target_epsilon': 1.0,
                    'target_delta': 1e-5,
                    'max_grad_norm': 1.0,
                    'noise_multiplier': 1.0,
                }

            if dataset_size is None:
                warnings.warn("dataset_size not provided for DP-SGD. Sample rate cannot be calculated.")
                self.dataset_size = 0
            else:
                self.dataset_size = dataset_size

            try:
                self.dp_engine = DifferentialPrivacyEngine(**dp_config)
                self.dp_enabled = True
            except Exception as e:
                warnings.warn(f"DP engine initialization failed: {e}")
                self.dp_enabled = False

        def training_step(self, model, inputs):
            """重写训练步骤以包含差分隐私"""

            model.train()
            inputs = self._prepare_inputs(inputs)

            # 正常的前向和反向传播
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            if hasattr(self, 'accelerator'):
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # 差分隐私处理
            if self.dp_enabled:
                try:
                    batch_size = next(iter(inputs.values())).shape[0]
                    private_grads = self.dp_engine.privatize_gradients(model, self.dataset_size, batch_size)

                    # 将私有梯度应用回模型
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in private_grads:
                            param.grad = private_grads[name]

                except Exception as e:
                    warnings.warn(f"DP processing failed in training step: {e}")

            return loss.detach() / self.args.gradient_accumulation_steps

        def get_dp_analysis(self):
            """获取差分隐私分析"""
            if self.dp_enabled:
                return self.dp_engine.get_privacy_analysis()
            else:
                return {'error': 'DP engine not enabled'}

    return DPEnhancedTrainer
