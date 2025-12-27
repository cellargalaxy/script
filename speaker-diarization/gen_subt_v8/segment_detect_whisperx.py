import util
import whisperx
import tool_whisperx

logger = util.get_logger()

model = None


def get_model():
    global model
    if model:
        return model
    device = util.get_device_type()
    compute_type = util.get_compute_type()
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    return model


def exec_gc():
    global model
    model = None
    util.exec_gc()


def transcribe(audio, language=None):
    model = get_model()
    segments, language = tool_whisperx.transcribe(model, audio, language=language)
    return segments, language
