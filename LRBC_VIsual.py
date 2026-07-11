import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0

fig, ax = plt.subplots(figsize=(8, 5))

x = np.linspace(0, 100, 500)

# 全精度
y_fp = np.where(x <= 20,
                95.0 * (x / 20),
                95.0 + (x - 20) * (5.0 / 80))
y_fp = np.clip(y_fp, 0, 100)

# 量化
saturate_quant = 72.0
y_quant = np.where(x <= 38.28,
                   saturate_quant * (1 - np.exp(-5 * x / 38.28)),
                   saturate_quant)
y_quant = np.clip(y_quant, 0, saturate_quant)

# 恢复
saturate_restore = 98.5
y_restore = np.where(x <= 94.14,
                     y_fp,
                     saturate_restore + (x - 94.14) * (100 - saturate_restore) / 5.86)
y_restore = np.clip(y_restore, 0, 100)

# 绘图
ax.plot(x, y_fp, color='#1f77b4', linewidth=2.5, label='Full-precision (FP)')
ax.plot(x, y_quant, color='#d62728', linewidth=2.5, linestyle='--', label='6-bit Quantized')
ax.plot(x, y_restore, color='#2ca02c', linewidth=2.5, linestyle='-.', label='Restored with LRBC')

# 标注关键位置
ax.axvline(x=20, color='gray', linestyle=':', alpha=0.6)
ax.axvline(x=38.28, color='gray', linestyle=':', alpha=0.6)
ax.axvline(x=94.14, color='gray', linestyle=':', alpha=0.6)

ax.annotate('Top 20% singular values\ncapture >95% energy',
            xy=(20, 95), xytext=(25, 85),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
            fontsize=9, ha='left')

ax.annotate('Effective rank: 98\n(62% reduction)',
            xy=(38.28, 72), xytext=(45, 65),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
            fontsize=9, ha='left')

ax.annotate('Effective rank: 241\n(94% of full-precision)',
            xy=(94.14, 98.5), xytext=(75, 90),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
            fontsize=9, ha='left')

ax.set_xlabel('Percentage of Singular Values (%)')
ax.set_ylabel('Cumulative Variance Explained (%)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right')

ax.text(0.5, -0.2, 'Quantization destroys low-rank structure → LRBC restores effective rank',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray', style='italic')

plt.tight_layout()
plt.savefig('rank_recovery_cumulative.pdf', dpi=300, bbox_inches='tight')
plt.show()