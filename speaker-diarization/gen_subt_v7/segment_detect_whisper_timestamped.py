import whisper_timestamped as whisper
import util
import tool_whisper_timestamped

logger = util.get_logger()

model = None


def get_model():
    global model
    if model:
        return model
    model = whisper.load_model("large-v3", device=util.get_device_type())
    return model


def exec_gc():
    global model
    model = None
    util.exec_gc()


def transcribe(audio):
    model = get_model()
    segments = tool_whisper_timestamped.transcribe(model, audio)
    return segments
