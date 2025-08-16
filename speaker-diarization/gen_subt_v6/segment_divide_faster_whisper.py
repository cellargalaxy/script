import util_audio
import math
from faster_whisper import WhisperModel
import util
import util_subt

logger = util.get_logger()

model = None


def get_model():
    global model
    if model:
        return model
    model = WhisperModel("large-v3", device=util.get_device_type(), compute_type=util.get_compute_type())
    return model


def exec_gc():
    global model
    model = None
    util.exec_gc()


def segment_divide(audio):
    last_end = len(audio)

    samples = util_audio.pydub2faster_whisper(audio)
    model = get_model()
    results, info = model.transcribe(samples)

    segments = []
    for result in results:
        start = math.floor(result.start * 1000)
        if start < 0:
            start = 0
        end = math.ceil(result.end * 1000)
        if last_end < end:
            end = last_end
        segments.append({"start": start, "end": end, "text": result.text})

    segments = util_subt.clipp_segments(segments, last_end)

    return segments
