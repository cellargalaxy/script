import numpy as np


def pydub2wavfile(data):
    raw_data = data.raw_data
    simple_rate = data.frame_rate
    channels = data.channels
    sample_width = data.sample_width
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError("pydub转wavfile，非法sample_width: {}".format(sample_width))
    audio_np = np.frombuffer(raw_data, dtype=dtype)
    if channels > 1:
        audio_np = audio_np.reshape((-1, channels))
    return simple_rate, audio_np
