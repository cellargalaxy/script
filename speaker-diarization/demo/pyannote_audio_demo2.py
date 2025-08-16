import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import torch
from pyannote.audio import Pipeline


def seconds_to_srt_time(seconds):
    from datetime import timedelta
    td = timedelta(seconds=seconds)
    return str(td)[:11].replace(".", ",").zfill(12)


pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda"))
diarization = pipeline("../gen_subt_v5/output/long/part_split/split/00000_speech.wav")

with open("output.srt", "w") as f:
    for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True), 1):
        start_time = seconds_to_srt_time(turn.start)
        end_time = seconds_to_srt_time(turn.end)
        f.write(f"{idx}\n{start_time} --> {end_time}\n{speaker}\n\n")
