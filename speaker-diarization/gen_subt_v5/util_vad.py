import util_audio
from ten_vad import TenVad
import math


def has_speech_by_data(data, frame_rate=50, threshold=0.8):
    simple_rate, data = util_audio.pydub2wavfile(data)
    frame_simple = int(simple_rate / frame_rate)
    frame_cnt = data.shape[0] // frame_simple
    ten_vad_instance = TenVad(frame_simple)
    max_probability = 0
    probability_ms = 0
    for i in range(frame_cnt):
        audio_data = data[i * frame_simple: (i + 1) * frame_simple]
        probability, _ = ten_vad_instance.process(audio_data)
        mean = math.floor((i + 0.5) * (1.0 / frame_rate) * 1000)
        if max_probability < probability:
            max_probability = probability
            probability_ms = mean
    return threshold <= max_probability, max_probability, probability_ms


def has_silene_by_data(data, frame_rate=50, threshold=0.6):
    simple_rate, data = util_audio.pydub2wavfile(data)
    frame_simple = int(simple_rate / frame_rate)
    frame_cnt = data.shape[0] // frame_simple
    ten_vad_instance = TenVad(frame_simple)
    min_probability = 1
    probability_ms = 0
    for i in range(frame_cnt):
        audio_data = data[i * frame_simple: (i + 1) * frame_simple]
        probability, _ = ten_vad_instance.process(audio_data)
        mean = math.floor((i + 0.5) * (1.0 / frame_rate) * 1000)
        if probability < min_probability:
            min_probability = probability
            probability_ms = mean
    return min_probability <= threshold, min_probability, probability_ms
