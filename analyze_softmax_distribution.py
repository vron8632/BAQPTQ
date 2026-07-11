"""
统计和分析Softmax后分布的脚本
基于AIQVIT/visualization/vis.py中的代码
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def analyze_softmax_distribution(attn_data_path='attn.npy'):
    """
    分析Softmax后的分布特征
    """
    if not os.path.exists(attn_data_path):
        print(f"警告: 找不到文件 {attn_data_path}，生成模拟数据进行演示")
        # 生成模拟的Softmax输出数据
        torch.manual_seed(42)
        # 模拟注意力分数的幂律分布
        x = torch.rand(10000)
        # 使用幂律变换模拟注意力分布
        x = torch.pow(x, 5)  # 使分布偏向0
        x = x / x.sum(dim=-1, keepdim=True)  # 确保和为1
        attn = x.numpy()
    else:
        attn = np.load(attn_data_path).flatten()
    
    print(f"数据统计:")
    print(f"  数据点数量: {len(attn)}")
    print(f"  最小值: {attn.min():.8f}")
    print(f"  最大值: {attn.max():.6f}")
    print(f"  平均值: {attn.mean():.6f}")
    print(f"  中位数: {attn.median() if hasattr(attn, 'median') else np.median(attn):.6f}")
    
    # 分析分布特性
    small_values_ratio = np.sum(attn < 0.01) / len(attn)
    medium_values_ratio = np.sum((attn >= 0.01) & (attn < 0.1)) / len(attn)
    large_values_ratio = np.sum(attn >= 0.1) / len(attn)
    
    print(f"\n分布分析:")
    print(f"  小于0.01的值比例: {small_values_ratio:.2%}")
    print(f"  [0.01, 0.1) 区间值比例: {medium_values_ratio:.2%}")
    print(f"  大于等于0.1的值比例: {large_values_ratio:.2%}")
    
    # 绘制直方图
    plt.figure(figsize=(12, 5))
    
    # 子图1: 整体分布
    plt.subplot(1, 2, 1)
    plt.hist(attn, bins=100, density=False, color='blue', alpha=0.5)
    plt.title('Softmax Output Distribution (Overall)')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.xlim(0, 1)
    
    # 子图2: 关注小值区域（对数刻度）
    plt.subplot(1, 2, 2)
    # 只显示较小的值以更好地观察分布
    attn_small = attn[attn < 0.1]
    if len(attn_small) > 0:
        plt.hist(attn_small, bins=100, density=False, color='red', alpha=0.5)
        plt.title('Softmax Output Distribution (Values < 0.1)')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.xlim(0, 0.1)
    
    plt.tight_layout()
    plt.show()
    
    return attn


def analyze_agq_effect(attn_data, taus=[0.5, 1.0, 2.0, 4.0]):
    """
    分析不同tau值对AGQ量化的影响
    """
    print(f"\nAGQ量化分析 (不同tau值):")
    print(f"{'Tau':<8} {'Max Error':<12} {'Mean Error':<12} {'Std Error':<12}")
    print("-" * 50)
    
    x = torch.tensor(attn_data, dtype=torch.float32)
    
    for tau in taus:
        # AGQ量化过程
        x_clamped = torch.clamp(x, 1e-20, None)
        scale = x.max()  # 使用最大值作为scale
        
        # AGQ量化公式: x_q = scale * 2^(-x_int/tau)
        # 反过来: x_int = -tau * log2(x/scale)
        x_int = torch.round(-tau * torch.log2(x_clamped / scale))
        x_q = torch.clamp(x_int, 0, 255)  # 8-bit
        x_dequant = scale * (2 ** (-x_q / tau))
        
        # 计算量化误差
        errors = torch.abs(x - x_dequant)
        max_error = errors.max().item()
        mean_error = errors.mean().item()
        std_error = errors.std().item()
        
        print(f"{tau:<8.1f} {max_error:<12.6f} {mean_error:<12.6f} {std_error:<12.6f}")


def compare_quantization_methods(attn_data):
    """
    比较不同的量化方法
    """
    print(f"\n量化方法比较:")
    print(f"{'Method':<15} {'Max Error':<12} {'Mean Error':<12} {'Std Error':<12}")
    print("-" * 55)
    
    x = torch.tensor(attn_data, dtype=torch.float32)
    
    # 1. 均匀量化 (8-bit)
    scale_uniform = x.max()
    x_int_uniform = torch.round(x / (scale_uniform / 255.0))
    x_int_uniform = torch.clamp(x_int_uniform, 0, 255)
    x_dequant_uniform = x_int_uniform * (scale_uniform / 255.0)
    
    errors_uniform = torch.abs(x - x_dequant_uniform)
    print(f"{'Uniform':<15} {errors_uniform.max().item():<12.6f} "
          f"{errors_uniform.mean().item():<12.6f} {errors_uniform.std().item():<12.6f}")
    
    # 2. AGQ量化 (tau=2.0)
    tau = 2.0
    x_clamped = torch.clamp(x, 1e-20, None)
    scale_agq = x.max()
    
    x_int_agq = torch.round(-tau * torch.log2(x_clamped / scale_agq))
    x_q_agq = torch.clamp(x_int_agq, 0, 255)
    x_dequant_agq = scale_agq * (2 ** (-x_q_agq / tau))
    
    errors_agq = torch.abs(x - x_dequant_agq)
    print(f"{'AGQ (tau=2)':<15} {errors_agq.max().item():<12.6f} "
          f"{errors_agq.mean().item():<12.6f} {errors_agq.std().item():<12.6f}")


def main():
    print("=== Softmax输出分布分析 ===")
    
    # 分析Softmax分布
    attn_data = analyze_softmax_distribution()
    
    # 分析不同tau值的AGQ效果
    analyze_agq_effect(attn_data)
    
    # 比较不同量化方法
    compare_quantization_methods(attn_data)
    
    print(f"\n分析完成！")


if __name__ == "__main__":
    main()