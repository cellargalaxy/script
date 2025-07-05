import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'



import torch
import numpy as np
import os

from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", use_auth_token=os.environ.get('auth_token', ''))

from pyannote.audio import Inference
inference = Inference(model, window="whole")
inference.to(torch.device("cuda"))
embedding1 = inference("../aaa/output/long/segment_split/00001_speech.wav")
embedding2 = inference("../aaa/output/long/segment_split/00012_speech.wav")

# 保证为二维
embedding1 = np.array(embedding1).reshape(1, -1)
embedding2 = np.array(embedding2).reshape(1, -1)

from scipy.spatial.distance import cdist
distance = cdist(embedding1, embedding2, metric="cosine")[0, 0]

print(f"相似度: {distance:.4f}")
