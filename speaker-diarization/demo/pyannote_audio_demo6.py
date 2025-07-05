import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
import util
import json

# 示例置信度数组（你可以换成自己的数据）
content = util.read_file('pyannote_audio_demo5.json', '[]')
confidences = json.loads(content)
confidences = np.clip(confidences, 0, 1)  # 保证在 [0,1] 区间内
confidences.sort()

# 构造 CDF 曲线（X 为置信度，Y 为累计比例）
x = confidences
y = np.linspace(0, 1, len(confidences))  # 累积分布函数

# 使用 KneeLocator 查找“拐点”
knee = KneeLocator(x, y, curve='concave', direction='increasing')

# 输出拐点置信度
print(f"拐点置信度阈值：{knee.knee}")

# 可视化
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='sum CDF')
if knee.knee:
    plt.axvline(knee.knee, color='red', linestyle='--', label=f'turn ≈ {knee.knee:.3f}')
plt.title("pyannote_audio_demo6.py")
plt.xlabel("confidence")
plt.ylabel("sum")
plt.legend()
plt.grid(True)
plt.show()
