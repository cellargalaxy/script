import matplotlib.pyplot as plt
import util
import torch
import numpy as np
import os
import util
from pyannote.audio import Model
from scipy.spatial.distance import cdist
from pyannote.audio import Inference
import json

inference = None
embedding_map = {}


def get_inference(auth_token):
    global inference
    if inference:
        return inference
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=auth_token)
    inference = Inference(model, window="whole")
    inference.to(torch.device(util.get_device_type()))
    return inference


def get_embedding(path, auth_token):
    global embedding_map
    embedding = embedding_map.get(path, None)
    if embedding is not None:
        return embedding
    inference = get_inference(auth_token)
    embedding = inference(path)
    embedding = np.array(embedding).reshape(1, -1)
    embedding_map[path] = embedding
    return embedding


def confidence_detect(path_i, path_j, auth_token):
    embedding_i = get_embedding(path_i, auth_token)
    embedding_j = get_embedding(path_j, auth_token)
    distance = cdist(embedding_i, embedding_j, metric="cosine")[0, 0]
    confidence = 1 - distance
    return confidence


dir = '../gen_subt_v4/output/long_jpn/segment_split'
files = util.listdir(dir)
files = sorted([s for s in files if "speech.wav" in s])  # 排序保证顺序一致

confidence_list = []
for i in range(len(files)):
    if i + 10 >= len(files):
        break

    file_path = os.path.join(dir, files[i])
    next_files = files[i + 1:i + 11]
    next_paths = [os.path.join(dir, nf) for nf in next_files]

    confidences = []
    # print("当前文件:", file_path)
    for next_path in next_paths:
        # print("  对比文件:", next_path)
        confidence = confidence_detect(file_path, next_path, '')
        confidences.append(confidence)
    confidence_list.append(confidences)

Z = np.array(confidence_list)
Z = np.sort(Z, axis=1)
Z = Z.T

# # 创建图像
# plt.figure(figsize=(15, 3))  # 宽一点好看

# # 使用imshow画热力图
# plt.imshow(Z, cmap='viridis', aspect='auto', origin='lower', vmin=0, vmax=1)

# # 添加颜色条（表示置信度）
# plt.colorbar(label='Confidence (Z)')

# # 设置坐标轴标签
# plt.xlabel('X (100 grid cells)')
# plt.ylabel('Y (10 grid cells)')
# plt.title('Confidence Heatmap')

# # 显示图像
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D画图必备
from matplotlib import cm

fig = plt.figure(figsize=(18, 8))
ax = fig.add_subplot(111, projection='3d')

# X轴：当前段索引，长度约284
# Y轴：后续10段索引，长度10
_x = np.arange(Z.shape[1])  # 284
_y = np.arange(Z.shape[0])  # 10
_xx, _yy = np.meshgrid(_x, _y)

x = _xx.ravel()
y = _yy.ravel()
z = np.zeros_like(x)  # 所有柱子底部高度为0

dx = dy = 0.8  # 每个柱子底面积大小（正方形）
dz = Z.ravel()  # 柱子高度，拉平成一维

# 颜色根据高度映射
norm = plt.Normalize(dz.min(), dz.max())
colors = cm.viridis(norm(dz))

ax.bar3d(x, y, z, dx, dy, dz, color=colors, shade=True)

# 设置坐标轴标签
ax.set_xlabel('Current segment index')
ax.set_ylabel('Next 10 segments')
ax.set_zlabel('Confidence')

# 设置标题
ax.set_title('3D Bar plot of Confidence')

plt.show()
