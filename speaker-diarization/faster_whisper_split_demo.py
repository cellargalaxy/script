# 可用

from faster_whisper import WhisperModel
import pysubs2
from pydub import AudioSegment
import os
from datetime import datetime

audio_file = "short.wav"
output_dir = "faster_whisper_split_demo"
device = "cpu"  # cuda/cpu
batch_size = 16
compute_type = "int8"
whisper_size = "large-v3"

print("start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
segments, info = model.transcribe(audio_file, beam_size=batch_size)

results = []
for s in segments:
    segment_dict = {'start': s.start, 'end': s.end, 'text': s.text}
    results.append(segment_dict)

subs = pysubs2.load_from_whisper(results)
subs.save('short.srt')

audio = AudioSegment.from_wav(audio_file)
for i, segment in enumerate(results):
    print(i, segment['start'], segment['end'], segment['text'])
    start_ms = int(segment['start'] * 1000)
    end_ms = int(segment['end'] * 1000)
    segment_audio = audio[start_ms:end_ms]
    segment_filename = os.path.join(output_dir, f"segment_{i + 1:03}.wav")
    segment_audio.export(segment_filename, format="wav")

print("end:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))