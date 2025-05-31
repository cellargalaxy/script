# 可用

import whisper_timestamped as whisper
import pysubs2
from whisperx.utils import get_writer
import json

audio = whisper.load_audio("../short.wav")

model = whisper.load_model("large-v3", device="cpu")

result = whisper.transcribe(model, audio)
print('result', json.dumps(result))

# vtt_writer = get_writer("vtt", ".")
# vtt_writer(
#     result,
#     "short.wav",
#     {"max_line_width": None, "max_line_count": None, "highlight_words": True},
# )

segments = result['segments']
results = []
for s in segments:
    segment_dict = {'start': s['start'], 'end': s['end'], 'text': s['text']}
    results.append(segment_dict)

print('results', json.dumps(results))

subs = pysubs2.load_from_whisper(results)
subs.save('short.srt')

whisper_result = {
    "segments": [],
    "word_segments": [],
    "language": result["language"],
}
for s in segments:
    ws = []
    words = s['words']
    for word in words:
        obj = {
            "word": word['text'],
            "start": word['start'],
            "end": word['end'],
            "score": word['confidence'],
            "speaker": '',
        }
        ws.append(obj)
        whisper_result['word_segments'].append(obj)

    segment = {
        "start": s['start'],
        "end": s['end'],
        "text": s['text'],
        "words": ws,
        "speaker": ""
    }
    whisper_result['segments'].append(segment)

print('whisper_result', json.dumps(whisper_result))
vtt_writer = get_writer("vtt", "..")
vtt_writer(
    whisper_result,
    "short.wav",
    {"max_line_width": None, "max_line_count": None, "highlight_words": True},
)
