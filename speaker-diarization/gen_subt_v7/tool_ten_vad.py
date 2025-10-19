from ten_vad import TenVad
import numpy as np
import util

logger = util.get_logger()


def pydub_ten_vad(audio):
    raw_data = audio.raw_data
    simple_rate = audio.frame_rate
    sample_width = audio.sample_width
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        logger.error(f"pydub转TenVad，非法sample_width: {sample_width}")
        raise ValueError(f"pydub转TenVad，非法sample_width: {sample_width}")
    np_data = np.frombuffer(raw_data, dtype=dtype)
    return simple_rate, np_data


def vad_confidence(data, frame_rate=50):
    simple_rate, np_data = pydub_ten_vad(data)
    frame_simple = int(simple_rate / frame_rate)
    frame_cnt = np_data.shape[0] // frame_simple
    ten_vad_instance = TenVad(frame_simple)
    confidences = []
    for i in range(frame_cnt):
        frame_data = np_data[i * frame_simple: (i + 1) * frame_simple]
        probability, _ = ten_vad_instance.process(frame_data)
        confidences.append(probability)
    return confidences




