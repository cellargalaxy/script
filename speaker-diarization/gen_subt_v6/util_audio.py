import numpy as np
import torch


def pydub2TenVad(audio):
    raw_data = audio.raw_data
    simple_rate = audio.frame_rate
    channels = audio.channels
    sample_width = audio.sample_width
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError("pydub转TenVad，非法sample_width: {}".format(sample_width))
    audio_np = np.frombuffer(raw_data, dtype=dtype)
    if channels > 1:
        audio_np = audio_np.reshape((-1, channels))
    return simple_rate, audio_np


def pydub2pyannote(audio):
    simple_rate = audio.frame_rate
    # 转成 numpy.int16 -> float32
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # 归一化到 -1~1
    samples /= np.iinfo(audio.array_type).max
    # 转成 torch.Tensor，形状 (channels, time)
    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)  # 单声道
    return simple_rate, waveform


def pydub2faster_whisper(audio):
    # 转 numpy 数组（float32，范围 -1.0 ~ 1.0）
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples
