import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn'

import whisperx
from pydub import AudioSegment
import numpy as np
import util
from faster_whisper import WhisperModel
import gc
import torch

audio_file = "/workspace/script/speaker-diarization/material/test.wav"
device = "cuda"  # cuda/cpu
batch_size = 16
compute_type = "float16"
whisper_size = "large-v3"


def pydub_faster_whisper(audio):
    # 转 numpy 数组（float32，范围 -1.0 ~ 1.0）
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples


audio = AudioSegment.from_wav(audio_file)
audio = pydub_faster_whisper(audio)

model = WhisperModel("large-v3", device=device, compute_type=compute_type)
results, info = model.transcribe(audio)
del model
gc.collect()
torch.cuda.empty_cache()

segments = []
for result in results:
    segments.append({"start": result.start, "end": result.end , "text": result.text})

print(info.language)
print(util.json_dumps(segments))

model_a, metadata = whisperx.load_align_model(language_code=info.language, device=device)
aligned_result = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)
del model_a
del metadata
gc.collect()
torch.cuda.empty_cache()
print('aligned_result', len(aligned_result['segments']), util.json_dumps(aligned_result))
