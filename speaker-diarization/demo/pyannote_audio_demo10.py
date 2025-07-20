import numpy as np
import matplotlib.pyplot as plt

# 构造一个 10 x 100 的随机 Z 值，表示置信度，范围 [0, 1]
# Y 是 10 行，X 是 100 列
Z = np.random.rand(10, 100)

# 创建图像
plt.figure(figsize=(15, 3))  # 宽一点好看

# 使用imshow画热力图
plt.imshow(Z, cmap='viridis', aspect='auto', origin='lower', vmin=0, vmax=1)

# 添加颜色条（表示置信度）
plt.colorbar(label='Confidence (Z)')

# 设置坐标轴标签
plt.xlabel('X (100 grid cells)')
plt.ylabel('Y (10 grid cells)')
plt.title('Confidence Heatmap')

# 显示图像
plt.show()
