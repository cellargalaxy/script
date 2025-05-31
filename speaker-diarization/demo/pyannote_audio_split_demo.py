import torch
from pyannote.audio import Pipeline
import json
from pydub import AudioSegment
import os
from datetime import datetime
from pathlib import Path

audio_file = "../short.wav"
device = "cpu"

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device(device))
diarization = pipeline(audio_file)

output_n1_dir = "pyannote_audio_split_demo"
i = 0
audio = AudioSegment.from_wav(audio_file)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start}s stop={turn.end}s speaker_{speaker}")
    start_ms = int(turn.start * 1000)
    end_ms = int(turn.end * 1000)
    segment_audio = audio[start_ms:end_ms]
    segment_filename = os.path.join(output_n1_dir, f"segment_{i + 1:03}.wav")
    segment_audio.export(segment_filename, format="wav")
    i = i + 1
