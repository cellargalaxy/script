from faster_whisper import WhisperModel
import util
import tool_faster_whisper

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


def transcribe(audio, language=None):
    model = get_model()
    segments, language = tool_faster_whisper.transcribe(model, audio, language=language)
    return segments, language
