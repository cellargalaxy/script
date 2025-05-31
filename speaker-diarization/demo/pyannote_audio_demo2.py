


import torch
from pyannote.audio import Pipeline

def seconds_to_srt_time(seconds):
    from datetime import timedelta
    td = timedelta(seconds=seconds)
    return str(td)[:11].replace(".", ",").zfill(12)

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cpu"))
diarization = pipeline("input.wav")

with open("output.srt", "w") as f:
    for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True), 1):
        start_time = seconds_to_srt_time(turn.start)
        end_time = seconds_to_srt_time(turn.end)
        f.write(f"{idx}\n{start_time} --> {end_time}\n{speaker}\n\n")