import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

from speechbrain.inference.diarization import Speech_Emotion_Diarization
import json

sed_model = Speech_Emotion_Diarization.from_hparams(source="speechbrain/emotion-diarization-wavlm-large",
                                                    run_opts={"device": "cuda"})
frame_class = sed_model.diarize_file("/workspace/script/speaker-diarization/gen_subt_v7/output/holo/split_audio/segment_divide_path/0069-苦手な食材は食べられないものじゃ得意なことは嘘を聞き分けることじゃ苦手なことは.wav")
print('frame_class', json.dumps(frame_class))
