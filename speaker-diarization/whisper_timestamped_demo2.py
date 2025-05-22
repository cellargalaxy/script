# 可用

import whisper_timestamped as whisper
import pysubs2
from whisperx.utils import get_writer
import json

audio = whisper.load_audio("short.wav")

model = whisper.load_model("large-v3", device="cpu")

result = whisper.transcribe(model, audio)

# vtt_writer = get_writer("vtt", ".")
# vtt_writer(
#     result,
#     "short.wav",
#     {"max_line_width": None, "max_line_count": None, "highlight_words": True},
# )

segments = result['segments']
print('segments', json.dumps(segments))
results = []
for s in segments:
    segment_dict = {'start': s['start'], 'end': s['end'], 'text': s['text']}
    results.append(segment_dict)

print('results', json.dumps(results))

subs = pysubs2.load_from_whisper(results)
subs.save('short.srt')
