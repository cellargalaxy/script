import os

os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

import torch
import nemo.collections.asr as nemo_asr

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

embs1 = speaker_model.get_embedding("../gen_subt_v3/output/long_jpn/segment_split/00000_speech.wav").squeeze()
embs2 = speaker_model.get_embedding("../gen_subt_v3/output/long_jpn/segment_split/00000_speech.wav").squeeze()
# Length Normalize
X = embs1 / torch.linalg.norm(embs1)
Y = embs2 / torch.linalg.norm(embs2)
# Score
similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
similarity_score = (similarity_score + 1) / 2
print('similarity_score', similarity_score)

