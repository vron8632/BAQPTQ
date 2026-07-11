import torch.nn as nn
import torch
import math

class MoRAConfig:
    """MoRA配置类 - 参考AIQVIT优化参数"""
    def __init__(self, 
                 enabled=True,
                 rank=64,           # 参考AIQVIT调整rank值
                 alpha=16,          # 参考AIQVIT调整alpha值
                 target_modules=None):
        self.enabled = enabled
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ['Linear']
        # bypass_ratio已废弃，使用直接相加方式

class TrueMoRALayer(nn.Module):
    """
    真正的MoRA层 - 基于分组的非参数化压缩与解压缩方法
    
    核心创新：
    1. 使用方阵M (r' × r') 替代LoRA的A、B矩阵
    2. 引入分组压缩算子Fcomp：k → r' (通过分组求平均)
    3. 引入分组解压算子Fdecomp：r' → d (通过复制组平均值)
    4. 参数量：r'^2 ≈ (d+k)×r，实现参数平衡
    """
    
    def __init__(self, in_dim, out_dim, lora_rank, alpha=16):  # 使用优化后的默认alpha值
        super().__init__()
        self.in_dim = in_dim  # k
        self.out_dim = out_dim  # d
        self.alpha = alpha
        self.lora_rank = lora_rank  # 原始LoRA的rank
        
        # 计算方阵维度：r' = sqrt((d+k)×r)
        self.square_rank = int(math.sqrt((in_dim + out_dim) * lora_rank))
        
        # 核心组件2: 高阶方阵 M (r' × r')
        self.square_matrix = nn.Parameter(torch.zeros(self.square_rank, self.square_rank))
        
        # 初始化策略 - 参考AIQVIT优化策略
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化参数 - 参考AIQVIT的初始化策略"""
        # 方阵：接近单位矩阵的小扰动（参考AIQVIT优化）
        std_square = 0.01 / math.sqrt(self.square_rank)  # 更小的初始标准差
        nn.init.normal_(self.square_matrix, 0, std_square)
        # 添加单位矩阵成分，保证初始时接近恒等变换
        with torch.no_grad():
            # 使用更小的初始值，避免初始阶段的过大影响
            self.square_matrix += 0.01 * torch.eye(self.square_rank, device=self.square_matrix.device)
    
    def forward(self, x):
        """
        MoRA前向传播 - 使用分组平均的压缩与解压缩方法
        
        计算路径：
        x (batch, seq, k) → Fcomp(分组求平均) → (batch, seq, r') → M → (batch, seq, r') → Fdecomp(组内复制) → (batch, seq, d)
        
        Args:
            x: 输入张量 (batch_size, seq_len, in_dim) 或其他多维张量
        
        Returns:
            MoRA增量输出，形状同输入
        """
        # 保存原始形状用于后续恢复
        original_shape = x.shape
        original_ndim = x.ndim
        batch_size_original = x.shape[0]
        
        # 将输入展平为 (batch, seq, in_dim) 的3维形式
        if x.ndim == 2:
            # (batch, in_dim) → (batch, 1, in_dim)
            x = x.unsqueeze(1)
            squeeze_needed = True
        elif x.ndim > 3:
            # (batch, h, w, c) → (batch, h*w, c)
            # 保存中间维度的形状
            spatial_shape = x.shape[1:-1]  # 获取 (h, w)
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1, x.shape[-1])
            squeeze_needed = False
        else:
            squeeze_needed = False
        
        batch_size, seq_len, in_dim = x.shape
        
        # Step 1: 分组压缩 - 将输入从k维压缩到r'维，通过分组求平均
        compressed = self._group_compress(x, self.square_rank)
        
        # Step 2: 高阶方阵变换 - 核心创新
        transformed = torch.matmul(compressed, self.square_matrix)  # (batch, seq, r')
        
        # Step 3: 分组解压 - 将结果从r'维解压到d维，通过复制组平均值
        output = self._group_decompress(transformed, self.out_dim)
        
        # Step 4: 应用alpha缩放
        result = self.alpha * output
        
        # 恢复原始形状
        if squeeze_needed:
            result = result.squeeze(1)  # 移除添加的维度
        elif original_ndim > 3:
            # (batch, seq, out_dim) → (batch, h, w, out_dim)
            result = result.reshape(batch_size_original, *spatial_shape, self.out_dim)
        
        return result
    
    def _group_compress(self, x, target_dim):
        """
        分组压缩：将输入从原始维度压缩到目标维度（GPU优化版本）
        x: (batch_size, seq_len, in_dim)
        return: (batch_size, seq_len, target_dim)
        """
        batch_size, seq_len, in_dim = x.shape
        
        # 如果目标维度大于等于输入维度，直接扩展或截断
        if target_dim >= in_dim:
            if target_dim == in_dim:
                return x
            else:
                # 扩展：复制最后的特征直到达到目标维度
                expanded_x = torch.zeros(batch_size, seq_len, target_dim, device=x.device, dtype=x.dtype)
                expanded_x[:, :, :in_dim] = x
                if target_dim > in_dim:
                    expanded_x[:, :, in_dim:] = x[:, :, -1:].expand(-1, -1, target_dim - in_dim)
                return expanded_x
        
        # 否则执行压缩（使用张量操作，避免Python循环）
        group_size = math.ceil(in_dim / target_dim)
        
        # 填充到可以整除的长度
        padded_dim = target_dim * group_size
        if padded_dim > in_dim:
            # 填充零到末尾
            padding = torch.zeros(batch_size, seq_len, padded_dim - in_dim, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x[:, :, :padded_dim]
        
        # 重塑并求平均：(batch, seq, target_dim*group_size) -> (batch, seq, target_dim, group_size) -> (batch, seq, target_dim)
        compressed = x_padded.reshape(batch_size, seq_len, target_dim, group_size).mean(dim=-1)
        
        return compressed
    
    def _group_decompress(self, x, target_dim):
        """
        分组解压：将输入从当前维度解压到目标维度（GPU优化版本）
        x: (batch_size, seq_len, in_dim)
        return: (batch_size, seq_len, target_dim)
        """
        batch_size, seq_len, in_dim = x.shape
        
        # 如果目标维度小于等于输入维度，直接截断
        if target_dim <= in_dim:
            return x[:, :, :target_dim]
        
        # 否则执行解压缩（使用张量操作，避免Python循环）
        group_size = math.ceil(target_dim / in_dim)
        
        # 使用repeat_interleave实现快速复制：(batch, seq, in_dim) -> (batch, seq, in_dim*group_size)
        output = x.repeat_interleave(group_size, dim=-1)
        
        # 截断到目标维度
        output = output[:, :, :target_dim]
        
        return output
    
    def get_effective_rank(self):
        """计算MoRA的有效秩"""
        with torch.no_grad():
            U, S, V = torch.svd(self.square_matrix)
            threshold = 1e-6
            effective_rank = torch.sum(S > threshold).item()
            return effective_rank, S.max().item(), S.min().item()
    
    def get_parameter_stats(self):
        """获取参数统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'total_params': total_params,
            'square_matrix_params': self.square_matrix.numel(),
            'square_rank': self.square_rank,
        }