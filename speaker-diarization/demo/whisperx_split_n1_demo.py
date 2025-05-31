import whisperx
from whisperx.utils import get_writer
import json
from pydub import AudioSegment
import os
from datetime import datetime
from pathlib import Path

API_TOKEN = ""
audio_file = "../short.wav"
device = "cpu"
batch_size = 16
compute_type = "int8"
whisper_size = "large-v3"

print("start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

model = whisperx.load_model(whisper_size, device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print('result', json.dumps(result))

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print('aligned_result', json.dumps(aligned_result))

diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=API_TOKEN, device=device)
diarize_segments = diarize_model(audio)
diarize_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
print('diarize_result', json.dumps(diarize_result))

diarize_result["language"] = result["language"]
vtt_writer = get_writer("vtt", "..")
vtt_writer(
    diarize_result,
    audio_file,
    {"max_line_width": None, "max_line_count": None, "highlight_words": True},
)


output_n1_dir = "whisperx_split_demo_n1"

audio = AudioSegment.from_wav(audio_file)
for i, segment in enumerate(aligned_result["segments"]):
    print(i, segment['start'], segment['end'], segment['text'])
    start_ms = int(segment['start'] * 1000)
    end_ms = int(segment['end'] * 1000)
    segment_audio = audio[start_ms:end_ms]
    segment_filename = os.path.join(output_n1_dir, f"segment_{i + 1:03}.wav")
    segment_audio.export(segment_filename, format="wav")


print("end:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))