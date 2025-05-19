from faster_whisper import WhisperModel
import pysubs2

audio_file = "short.mkv"
device = "cpu"  # cuda/cpu
batch_size = 16
compute_type = "int8"
whisper_size = "large-v3"

model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
segments, info = model.transcribe(audio_file, beam_size=batch_size)

results = []
for s in segments:
    segment_dict = {'start': s.start, 'end': s.end, 'text': s.text}
    results.append(segment_dict)

subs = pysubs2.load_from_whisper(results)
subs.save('short.srt')
