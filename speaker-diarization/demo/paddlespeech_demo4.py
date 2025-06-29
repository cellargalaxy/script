import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

from speechbrain.inference.diarization import Speech_Emotion_Diarization
import json

sed_model = Speech_Emotion_Diarization.from_hparams(source="speechbrain/emotion-diarization-wavlm-large",
                                                    savedir='tmpdir', run_opts={"device": "cuda"})
frame_class = sed_model.diarize_file("../demo_eng_single.wav")
print('frame_class', json.dumps(frame_class))
