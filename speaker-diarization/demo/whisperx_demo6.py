# https://github.com/m-bain/whisperX/issues/692#issuecomment-1992646967

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn'

import whisperx
from whisperx.utils import get_writer
import json
import gc
import torch

API_TOKEN = ""
audio_file = "../long_jpn.wav"
device = "cuda"  # cuda/cpu
batch_size = 16
compute_type = "float16"
whisper_size = "large-v3"

model = whisperx.load_model(whisper_size, device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print('result', json.dumps(result))
gc.collect()
torch.cuda.empty_cache()
del model

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print('aligned_result',len(aligned_result['segments']), json.dumps(aligned_result))
gc.collect()
torch.cuda.empty_cache()
del model_a

diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=API_TOKEN, device=device)
diarize_segments = diarize_model(audio)
diarize_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
diarize_result["language"] = result["language"]
print('diarize_result', json.dumps(diarize_result))

vtt_writer = get_writer("vtt", "..")
vtt_writer(
    diarize_result,
    audio_file,
    {"max_line_width": None, "max_line_count": None, "highlight_words": True},
)
