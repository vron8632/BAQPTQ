import torch
from ptq4sam.quantization.dynamic_interval_quantizer import DynamicIntervalAdaptiveGranularityQuantizer

# 创建一个测试用的Softmax输出张量
torch.manual_seed(42)
x = torch.rand(5, 10)  # 小张量用于测试
x = torch.softmax(x, dim=-1)  # 模拟Softmax输出
print(f"输入张量形状: {x.shape}")
print(f"输入张量范围: [{x.min():.6f}, {x.max():.6f}]")
print(f"输入张量总和 (沿最后一维): {x.sum(dim=-1)}")

# 创建量化器实例
quantizer = DynamicIntervalAdaptiveGranularityQuantizer(
    n_bits=8,
    low_org=0.0,  # 设置为0
    enable_interval_learning=True
)

print(f"low_org: {quantizer.low_org}")
print(f"初始 mid_raw: {quantizer.mid_raw.item():.4f}")
print(f"初始 tau_critical_raw: {quantizer.tau_critical_raw.item():.4f}")

try:
    # 进行量化 - 这会触发初始化
    x_quant = quantizer(x)
    print(f"量化成功!")
    print(f"量化后张量范围: [{x_quant.min():.6f}, {x_quant.max():.6f}]")
    print(f"mid值: {quantizer.get_mid().item():.4f}")
    print(f"tau值: {quantizer.get_tau_critical().item():.4f}")
    
    # 再次量化以确保一致性
    x_quant2 = quantizer(x)
    print(f"第二次量化成功!")
    print(f"第二次量化后张量范围: [{x_quant2.min():.6f}, {x_quant2.max():.6f}]")
    
except Exception as e:
    print(f"量化失败: {e}")
    import traceback
    traceback.print_exc()