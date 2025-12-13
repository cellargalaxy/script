import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

from voicefixer import VoiceFixer, Vocoder

voicefixer = VoiceFixer()
voicefixer.restore(
    input="/workspace/script/speaker-diarization/material/006.wav",
    output="/workspace/script/speaker-diarization/material/006_restore.wav",
    cuda=True,
    mode=0,
)
vocoder = Vocoder(sample_rate=44100)
vocoder.oracle(
    fpath="/workspace/script/speaker-diarization/material/006.wav",
    out_path="/workspace/script/speaker-diarization/material/006_oracle.wav",
    cuda=True,
)
