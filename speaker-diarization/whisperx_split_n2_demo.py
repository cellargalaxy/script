import whisperx
from whisperx.utils import get_writer
import json
from pydub import AudioSegment
import os
from datetime import datetime
from pathlib import Path

API_TOKEN = ""
device = "cpu"
batch_size = 16
compute_type = "int8"
whisper_size = "large-v3"

print("start:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

output_n1_dir = "whisperx_split_demo_n1"
folder_path = Path(output_n1_dir)
for file in folder_path.rglob('*'):
    if not file.is_file():
        continue
    audio_file = file
    model = whisperx.load_model(whisper_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=API_TOKEN, device=device)
    diarize_segments = diarize_model(audio)
    diarize_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

    diarize_result["language"] = result["language"]
    vtt_writer = get_writer("vtt", "whisperx_split_demo_n1")
    vtt_writer(
        diarize_result,
        audio_file,
        {"max_line_width": None, "max_line_count": None, "highlight_words": True},
    )
    print(file, len(aligned_result["segments"]))

print("end:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
