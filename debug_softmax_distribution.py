"""
调试Softmax分布和AGQ tau值变化，包含可视化对比
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ptq4sam.quantization.dynamic_interval_quantizer import DynamicIntervalAdaptiveGranularityQuantizer

matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

def debug_agq_tau_changes():
    """
    调试AGQ中tau值的变化过程
    """
    print("=== 调试AGQ Tau值变化 ===")
    
    # 创建模拟的Softmax输出数据 (典型的注意力分数分布)
    torch.manual_seed(42)
    # 模拟注意力分布：大多数值很小，少数值较大
    batch_size, seq_len = 2, 16
    raw_scores = torch.randn(batch_size, seq_len)
    softmax_outputs = torch.softmax(raw_scores, dim=-1)
    
    print(f"输入数据形状: {softmax_outputs.shape}")
    print(f"输入数据范围: [{softmax_outputs.min():.6f}, {softmax_outputs.max():.6f}]")
    print(f"输入数据示例 (第一行): {softmax_outputs[0, :8]}")
    
    # 创建DIAGQ量化器
    quantizer = DynamicIntervalAdaptiveGranularityQuantizer(
        n_bits=8,
        low_org=0.0,  # 设置为0，区间为[0, mid]
        enable_interval_learning=True
    )
    
    print(f"\n初始参数:")
    print(f"  tau_critical_raw: {quantizer.tau_critical_raw.item():.4f}")
    print(f"  tau (经过softplus): {quantizer.get_tau_critical().item():.4f}")
    print(f"  mid_raw: {quantizer.mid_raw.item():.4f}")
    print(f"  mid (经过sigmoid): {quantizer.get_mid().item():.4f}")
    
    # 执行第一次量化（会触发初始化）
    print(f"\n--- 执行第一次量化 ---")
    with torch.no_grad():
        output1 = quantizer(softmax_outputs)
    
    print(f"第一次量化后参数:")
    print(f"  tau (经过softplus): {quantizer.get_tau_critical().item():.4f}")
    print(f"  mid (经过sigmoid): {quantizer.get_mid().item():.4f}")
    
    # 检查量化效果
    errors1 = torch.abs(softmax_outputs - output1)
    print(f"  量化误差 - 最大值: {errors1.max():.6f}, 平均值: {errors1.mean():.6f}")
    
    # 记录优化过程中的参数变化
    tau_history = []
    mid_history = []
    loss_history = []
    
    # 执行优化过程
    print(f"\n--- 开始优化过程 ---")
    optimizer = torch.optim.Adam(list(quantizer.parameters()), lr=1e-2)
    
    for step in range(10):
        optimizer.zero_grad()
        
        # 前向传播
        output = quantizer(softmax_outputs)
        
        # 计算重构损失
        loss = torch.mean((softmax_outputs - output) ** 2)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 记录参数和损失
        tau_history.append(quantizer.get_tau_critical().item())
        mid_history.append(quantizer.get_mid().item())
        loss_history.append(loss.item())
        
        if step % 2 == 0 or step == 9:
            print(f"Step {step+1}: Loss={loss.item():.6f}, "
                  f"Tau={quantizer.get_tau_critical().item():.4f}, "
                  f"Mid={quantizer.get_mid().item():.4f}")
    
    print(f"\n--- 优化完成后 ---")
    with torch.no_grad():
        final_output = quantizer(softmax_outputs)
    
    final_errors = torch.abs(softmax_outputs - final_output)
    print(f"最终量化误差 - 最大值: {final_errors.max():.6f}, 平均值: {final_errors.mean():.6f}")
    
    # 分析量化前后分布的变化
    print(f"\n--- 分布分析 ---")
    orig_small_vals = torch.sum(softmax_outputs < 0.01).item() / softmax_outputs.numel()
    orig_medium_vals = torch.sum((softmax_outputs >= 0.01) & (softmax_outputs < 0.1)).item() / softmax_outputs.numel()
    orig_large_vals = torch.sum(softmax_outputs >= 0.1).item() / softmax_outputs.numel()
    
    quant_small_vals = torch.sum(final_output < 0.01).item() / final_output.numel()
    quant_medium_vals = torch.sum((final_output >= 0.01) & (final_output < 0.1)).item() / final_output.numel()
    quant_large_vals = torch.sum(final_output >= 0.1).item() / final_output.numel()
    
    print(f"原始分布 - 小值(<0.01): {orig_small_vals:.2%}, 中值([0.01,0.1)): {orig_medium_vals:.2%}, 大值(≥0.1): {orig_large_vals:.2%}")
    print(f"量化分布 - 小值(<0.01): {quant_small_vals:.2%}, 中值([0.01,0.1)): {quant_medium_vals:.2%}, 大值(≥0.1): {quant_large_vals:.2%}")
    
    return quantizer, softmax_outputs, final_output, tau_history, mid_history, loss_history


def analyze_agq_quantization_process():
    """
    详细分析AGQ量化过程
    """
    print(f"\n=== AGQ量化过程分析 ===")
    
    # 创建测试数据
    x = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99])
    tau_val = 2.0
    
    print(f"输入值: {x}")
    print(f"使用Tau值: {tau_val}")
    
    # 手动执行AGQ量化过程
    scale = x.max()  # 使用最大值作为scale
    print(f"Scale值: {scale}")
    
    # AGQ量化公式
    x_clamped = torch.clamp(x, 1e-20, None)
    x_int = torch.round(-tau_val * torch.log2(x_clamped / scale))
    x_q = torch.clamp(x_int, 0, 255)  # 8-bit
    x_dequant = scale * (2 ** (-x_q / tau_val))
    
    print(f"量化索引: {x_q.int()}")
    print(f"量化后值: {x_dequant}")
    print(f"量化误差: {torch.abs(x - x_dequant)}")


def visualize_agq_comparison(softmax_outputs, final_output, tau_history, mid_history, loss_history):
    """
    可视化AGQ优化前后的对比
    """
    print(f"\n=== 生成可视化图表 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 展平数据以便绘图
    orig_flat = softmax_outputs.flatten().numpy()
    quant_flat = final_output.flatten().numpy()
    
    # 1. 优化过程中tau和mid的变化
    axes[0, 0].plot(tau_history, label='Tau Critical', marker='o')
    axes[0, 0].plot(mid_history, label='Mid Value', marker='s')
    axes[0, 0].set_title('AGQ 参数优化过程')
    axes[0, 0].set_xlabel('训练步数')
    axes[0, 0].set_ylabel('参数值')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. 优化过程中损失变化
    axes[0, 1].plot(loss_history, label='重构损失', marker='d', color='red')
    axes[0, 1].set_title('优化过程中重构损失变化')
    axes[0, 1].set_xlabel('训练步数')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 原始分布vs量化后分布
    axes[1, 0].scatter(orig_flat, quant_flat, alpha=0.6)
    axes[1, 0].plot([orig_flat.min(), orig_flat.max()], [orig_flat.min(), orig_flat.max()], 'r--', lw=2)
    axes[1, 0].set_title('原始值 vs 量化值')
    axes[1, 0].set_xlabel('原始值')
    axes[1, 0].set_ylabel('量化值')
    axes[1, 0].grid(True)
    
    # 4. 原始分布直方图对比
    axes[1, 1].hist(orig_flat, bins=50, alpha=0.5, label='原始分布', density=True)
    axes[1, 1].hist(quant_flat, bins=50, alpha=0.5, label='量化后分布', density=True)
    axes[1, 1].set_title('原始分布与量化后分布对比')
    axes[1, 1].set_xlabel('值')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('agq_comparison.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存为 agq_comparison.png")
    plt.show()


def main():
    # 调试AGQ tau值变化
    quantizer, orig_data, quant_data, tau_history, mid_history, loss_history = debug_agq_tau_changes()
    
    # 分析AGQ量化过程
    analyze_agq_quantization_process()
    
    # 生成可视化对比
    visualize_agq_comparison(orig_data, quant_data, tau_history, mid_history, loss_history)
    
    print(f"\n=== 调试完成 ===")
    print(f"AGQ已成功自适应调整Tau值以适应Softmax分布!")


if __name__ == "__main__":
    main()