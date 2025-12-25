from ten_vad import TenVad
import numpy as np
import util
from pydub import AudioSegment

logger = util.get_logger()


def pydub_ten_vad(audio: AudioSegment):
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


def vad_confidence(audio: AudioSegment, frame_rate: int = 50):
    simple_rate, np_data = pydub_ten_vad(audio)
    frame_simple = int(simple_rate / frame_rate)
    window_ms = int(1000 / frame_rate)
    frame_cnt = np_data.shape[0] // frame_simple
    confidences = [0.0] * len(audio)
    instance = TenVad(frame_simple)
    for i in range(frame_cnt):
        sample_start = i * frame_simple
        sample_end = (i + 1) * frame_simple
        frame_data = np_data[sample_start:sample_end]
        confidence, _ = instance.process(frame_data)
        ms_start = i * window_ms
        ms_end = min((i + 1) * window_ms, len(audio))
        for ms in range(ms_start, ms_end):
            confidences[ms] = confidence
    return confidences
