import whisper_timestamped as whisper
from pydub import AudioSegment
import json
import os
from datetime import datetime

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

audio = whisper.load_audio("short.wav")

model = whisper.load_model("large-v3", device="cpu")

result = whisper.transcribe(model, audio)
print('result', json.dumps(result))

output_dir = "whisper_timestamped_spilt"

audio = AudioSegment.from_wav("short.wav")
segments = result['segments']
for i, segment in enumerate(segments):
    print(i, segment['start'], segment['end'], segment['text'])
    start_ms = int(segment['start'] * 1000)
    end_ms = int(segment['end'] * 1000)
    segment_audio = audio[start_ms:end_ms]
    segment_filename = os.path.join(output_dir, f"segment_{i + 1:03}.wav")
    segment_audio.export(segment_filename, format="wav")

print("end:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
