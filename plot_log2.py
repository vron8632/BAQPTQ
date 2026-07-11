import numpy as np
import matplotlib.pyplot as plt

# 生成x值，从0.01到10，避免0值因为log(0)未定义
x = np.linspace(0.01, 10, 400)

# 计算y = -log2(x)
y = -np.log2(x)

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$y = -\log_2(x)$', color='blue')

# 添加网格和标签
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of $y = -\log_2(x)$')
plt.legend()

# 设置坐标轴
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# 显示图形
plt.show()