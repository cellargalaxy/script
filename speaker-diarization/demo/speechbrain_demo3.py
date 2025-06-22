import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

# 不适合，识别不出某些人声，原因未明

from speechbrain.inference.VAD import VAD
import sub_util

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="model/pretrained_models/vad-crdnn-libriparty")
boundaries = VAD.get_speech_segments('../demo_eng_single.wav')

segments = []
for start, end in boundaries:
    segments.append({"start": int(start.item() * 1000), "end": int(end.item() * 1000), })
print(segments)
sub_util.save_segments_as_srt(segments, '../demo_eng_single.srt')
