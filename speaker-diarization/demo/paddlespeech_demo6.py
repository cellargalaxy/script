import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

wav_path = "/workspace/script/speaker-diarization/gen_subt_v7/output/holo/split_audio/segment_divide_path/0069-苦手な食材は食べられないものじゃ得意なことは嘘を聞き分けることじゃ苦手なことは.wav"

import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs =torchaudio.load('tests/samples/ASR/spk1_snt1.wav')
embeddings = classifier.encode_batch(signal)

unique_speakers = set()
for segment in boundaries:
    unique_speakers.add(segment[2])

num_speakers = len(unique_speakers)

if num_speakers == 0:
    print("没有检测到语音。")
elif num_speakers == 1:
    print("准确检测到：只有 1 个人在说话。")
else:
    print(f"准确检测到：有 {num_speakers} 个人在说话。")
