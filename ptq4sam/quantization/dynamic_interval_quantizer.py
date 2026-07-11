"""
动态区间自适应粒度量化器 (Dynamic Interval Adaptive Granularity Quantizer, DIAGQ)

融合AIQViT的DFQ(Dynamic Focused Quantizer)和PTQ4SAM的AGQ(Adaptive Granularity Quantization)
核心创新：
1. 使用DFQ学习最优关键区间 [0, mid]
2. 在关键区间内使用AGQ的幂次底数量化实现精细量化
3. 区间外使用粗粒度量化或截断
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from copy import deepcopy
from .util_quant import round_ste


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """L_p范数损失函数"""
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class DynamicIntervalAdaptiveGranularityQuantizer(nn.Module):
    """
    动态区间自适应粒度量化器 - 融合DFQ和AGQ的优势
    
    参数:
        n_bits: 量化比特数，默认8
        low_org: 初始下界，用于Softmax后分布
        tau_candidates: AGQ的tau候选值列表
        enable_interval_learning: 是否启用区间学习
    """
    
    def __init__(self, n_bits: int = 8, low_org: float = 0.0, 
                 tau_candidates=None, enable_interval_learning: bool = True):
        super(DynamicIntervalAdaptiveGranularityQuantizer, self).__init__()
        
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.low_org = low_org  # 现在是0
        self.enable_interval_learning = enable_interval_learning
        
        # DFQ区间参数 - 现在作为可学习参数
        self.register_buffer('inited', torch.zeros(1))
        # 将mid设为可学习参数，而不是从候选值中选择
        self.mid_raw = nn.Parameter(torch.tensor(0.2))  # 使用raw参数，通过sigmoid约束到[0,1]
        
        # AGQ粒度参数 - 现在作为可学习参数
        if tau_candidates is None:
            # 初始化为某个默认值，而不是候选列表
            self.tau_critical_raw = nn.Parameter(torch.tensor(0.693))  # log(2) ≈ 0.693，exp(0.693)≈2
        else:
            self.tau_critical_raw = nn.Parameter(torch.tensor(0.693))
        
        self.scale = nn.Parameter(torch.tensor(1.0))  # 也可设为可学习参数

    def get_mid(self):
        """获取mid值，通过sigmoid约束到[low_org, 1]范围"""
        import torch.nn.functional as F
        # 使用sigmoid将raw值约束到[0,1]，然后缩放到[low_org, 1]
        mid_normalized = F.sigmoid(self.mid_raw)
        mid = self.low_org + (1.0 - self.low_org) * mid_normalized
        return mid

    def get_tau_critical(self):
        """获取tau值，通过softplus确保为正值"""
        import torch.nn.functional as F
        return F.softplus(self.tau_critical_raw)

    def forward(self, x: torch.Tensor):
        """
        前向传播 - 完整版DIAGQ：动态区间 + 自适应粒度
        
        Args:
            x: Softmax输出，范围[0, 1]
        
        Returns:
            量化后的值
        """
        if self.inited == 0:
            self._init_quantization_params(x)
            self.inited.fill_(1)
        
        if self.scale is None:
            return x
        
        # 获取当前的可学习参数
        current_mid = self.get_mid()
        current_tau = self.get_tau_critical()
        
        # 根据配置选择模式
        if self.enable_interval_learning:
            # 完整版：使用动态区间策略
            x_quant = self._dynamic_interval_quantize_with_params(x, current_mid, current_tau)
        else:
            # 简化版：直接使用AGQ
            x_quant = self._agq_quantize(x, self.scale, current_tau)
        
        return x_quant
    
    def _init_quantization_params(self, x: torch.Tensor):
        """
        初始化量化参数：使用采样策略大幅减少显存占用
        关键优化：只使用小样本搜索，避免全量数据
        """
        # 关键优化：采样，避免处理全量数据
        if x.numel() > 10000:  # 如果数据量大于10000
            # 随机采样10000个点进行搜索
            indices = torch.randperm(x.numel(), device=x.device)[:10000]
            x_sample = x.flatten()[indices].reshape(-1)
        else:
            x_sample = x.clone().detach()
        
        if self.enable_interval_learning:
            # 完整版：联合搜索区间和tau - 但我们现在使用可学习参数
            # 可以使用搜索结果来初始化参数
            mid_candidates = [0.05, 0.1, 0.2]  # 搜索候选值用于初始化
            tau_candidates = [1, 2, 4]  # tau候选值用于初始化
            
            best_score = 1e+10
            best_mid = 0.2
            best_tau = 2
            best_scale = None
            
            # 使用采样数据的最大值作scale
            scale = x_sample.max()
            
            # 联合搜索 - 用于初始化可学习参数
            for mid in mid_candidates:
                for tau in tau_candidates:
                    with torch.no_grad():
                        x_q = self._test_quantize_with_interval(x_sample, scale, tau, mid)
                        score = lp_loss(x_sample, x_q, p=2, reduction='all')
                        del x_q
                    
                    if score < best_score:
                        best_score = score
                        best_mid = mid
                        best_tau = tau
                        best_scale = scale
            
            torch.cuda.empty_cache()
            
            # 初始化可学习参数
            with torch.no_grad():
                # 使用搜索结果初始化mid_raw参数 (通过反sigmoid)
                import torch.nn.functional as F
                if (1.0 - self.low_org) != 0:  # 避免除零
                    mid_normalized = (best_mid - self.low_org) / (1.0 - self.low_org)
                    # 确保mid_normalized是张量
                    mid_normalized = torch.tensor(mid_normalized).to(self.mid_raw.device)
                    mid_normalized = torch.clamp(mid_normalized, min=1e-6, max=1.0 - 1e-6)  # 避免边界值
                    self.mid_raw.copy_(torch.logit(mid_normalized))
                else:
                    self.mid_raw.copy_(torch.tensor(0.2))
                self.tau_critical_raw.copy_(torch.log(torch.tensor(best_tau)))  # 初始化tau
            self.scale.data.copy_(best_scale)
            
        else:
            # 简化版：只搜索tau - 使用采样数据
            tau_candidates = [1, 2, 4]  # tau候选值用于初始化
            best_score = 1e+10
            best_tau = 2
            best_scale = None
            
            scale = x_sample.max()
            
            for tau in tau_candidates:
                with torch.no_grad():
                    x_q = self._agq_quantize(x_sample, scale, tau)
                    score = lp_loss(x_sample, x_q, p=2, reduction='all')
                    del x_q
                
                if score < best_score:
                    best_score = score
                    best_tau = tau
                    best_scale = scale
            
            torch.cuda.empty_cache()
            
            # 初始化可学习参数
            with torch.no_grad():
                self.tau_critical_raw.copy_(torch.log(torch.tensor(best_tau)))  # 初始化tau
            self.scale.data.copy_(best_scale)

    def _test_quantize_with_interval(self, x: torch.Tensor, scale: torch.Tensor, tau: float, mid: float):
        """
        测试带区间的量化效果（用于初始化搜索）
        """
        low = self.low_org  # 现在是0
        
        # 分区域
        mask_critical = (x >= low) & (x <= mid)
        mask_below = x < low
        mask_above = x > mid
        
        x_quant = torch.zeros_like(x)
        
        # 关键区间：精细tau
        if mask_critical.any():
            x_quant[mask_critical] = self._agq_quantize(x[mask_critical], scale, tau)
        
        # 低值区：1.3倍 tau (不会发生，因为low=0)
        if mask_below.any():
            x_quant[mask_below] = self._agq_quantize(x[mask_below], scale, tau * 1.3)
        
        # 高值区：1.15倍 tau
        if mask_above.any():
            x_quant[mask_above] = self._agq_quantize(x[mask_above], scale, tau * 1.15)
        
        return x_quant
    
    def _quantize_with_params(self, x: torch.Tensor, mid: float, tau: float):
        """
        使用给定参数进行量化（用于搜索）
        
        Returns:
            (量化结果, scale参数)
        """
        # 确定关键区间
        low = self.low_org  # 现在是0
        
        # 计算区间内的分布特性
        in_critical_region = (x >= low) & (x <= mid)
        
        if in_critical_region.sum() == 0:
            # 没有值在关键区间内，使用简单量化
            delta = (mid - low) / self.n_levels
            x_int = round_ste((x - low) / delta)
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant * delta + low)
            return x_dequant, None
        
        # 使用全局最大值计算scale，确保所有值都能被正确量化
        # 这样可以保证不同区域的量化是连续的
        x_max = x.max()
        scale = x_max
        
        # 执行AGQ量化（应用到所有值）
        x_quant_full = self._agq_quantize(x, scale, tau)
        
        return x_quant_full, scale

    def _dynamic_interval_quantize_with_params(self, x: torch.Tensor, mid: torch.Tensor, tau: torch.Tensor):
        """
        使用可学习参数的动态区间量化：使用同一scale但不同tau来处理不同区域
        关键：使用保守的粒度倍数，避免过度粗糙化
        """
        # 确保区间有效
        mid_val = mid.item() if torch.is_tensor(mid) else mid
        low_val = self.low_org  # 现在是0

        # 分区域
        mask_critical = (x >= low_val) & (x <= mid_val)  # 关键区间 [0, mid]
        mask_below = x < low_val                      # 低值区 (空集，因为low=0)
        mask_above = x > mid_val                      # 高值区 (mid, 1]
        
        x_quant = torch.zeros_like(x)
        tau_val = tau.item() if torch.is_tensor(tau) else tau

        # 使用同一个scale保证连续性
        # 只调整tau来控制粒度
        
        # 1. 关键区间：使用最精细的tau
        if mask_critical.any():
            x_quant[mask_critical] = self._agq_quantize(
                x[mask_critical], self.scale, tau_val
            )
        
        # 2. 低值区：没有值会小于0，所以不需要处理
        
        # 3. 高值区：使用中等粒度的tau（保守的1.15倍）
        if mask_above.any():
            tau_medium = tau_val * 1.15  # 从1.5降低到1.15
            x_quant[mask_above] = self._agq_quantize(
                x[mask_above], self.scale, tau_medium
            )
        
        return x_quant

    def _dynamic_interval_quantize(self, x: torch.Tensor):
        """
        动态区间量化：使用同一scale但不同tau来处理不同区域
        关键：使用保守的粒度倍数，避免过度粗糙化
        """
        # 获取当前参数值
        mid = self.get_mid()
        tau = self.get_tau_critical()
        return self._dynamic_interval_quantize_with_params(x, mid, tau)

    def _agq_quantize(self, x: torch.Tensor, scale: torch.Tensor, tau: float):
        """
        自适应粒度量化（AGQ）- 使用幂次底数
        
        Args:
            x: 输入张量
            scale: 缩放参数
            tau: 粒度参数（底数的指数因子）
        
        Returns:
            量化后的张量
        """
        # 避免log(0)
        x = torch.clamp(x, 1e-20, None)
        
        # AGQ量化公式: x_q = scale * 2^(-x_int/tau)
        # 反过来: x_int = -tau * log2(x/scale)
        x_int = round_ste(-tau * (x / scale).log2())
        
        # 限制到量化范围
        x_q = torch.clamp(x_int, 0, self.n_levels - 1)
        
        # 反量化
        x_dequant = scale * (2 ** (-x_q / tau))
        
        # 处理可能的异常值
        x_dequant = torch.clamp(x_dequant, 0.0, 1.0)
        
        return x_dequant


class SimpleDFQQuantizer(nn.Module):
    """
    简化版DFQ量化器 - 仅使用动态区间，不使用AGQ
    适合快速实验和对比
    """
    
    def __init__(self, n_bits: int = 8, low_org: float = 0.0):
        super(SimpleDFQQuantizer, self).__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.low_org = low_org  # 现在是0
        self.register_buffer('inited', torch.zeros(1))
    
    def forward(self, x: torch.Tensor):
        if self.inited == 0:
            mid = self._init_quantization_scale(x)
            low = torch.tensor(self.low_org)
            self.mid = nn.Parameter(mid).contiguous()
            self.low = nn.Parameter(low)
            self.inited.fill_(1)
        
        # 滑动区间量化
        if self.mid > self.low and self.low > 0:
            delta_low = (self.mid - self.low) / self.n_levels
        else:
            delta_low = (self.mid - self.low_org) / self.n_levels
            self.low = nn.Parameter(torch.tensor(self.low_org))
        
        x_int_low = round_ste((x - self.low) / delta_low)
        x_quant_low = torch.clamp(x_int_low, 0, self.n_levels - 1)
        x_dequant_low = x_quant_low * delta_low + self.low
        
        return x_dequant_low
    
    def _init_quantization_scale(self, x: torch.Tensor):
        """搜索最优的mid值"""
        x_clone = x.clone().detach()
        cur_mid = -10
        best_score = 1e+10
        
        for mid in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.12, 0.2, 0.5, 1.0]:
            x_q = self._quantize(x_clone, mid)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                cur_mid = mid
        
        return torch.tensor(cur_mid)
    
    def _quantize(self, x, mid):
        """使用给定mid值量化"""
        delta_low = (mid - self.low_org) / self.n_levels
        x_int_low = round_ste((x - self.low_org) / delta_low)
        x_quant_low = torch.clamp(x_int_low, 0, self.n_levels - 1)
        x_dequant_low = x_quant_low * delta_low + self.low_org
        return x_dequant_low