import util
import whisperx
import tool_align_whisperx

logger = util.get_logger()

model_map = {}
metadata_map = {}


def get_model(language):
    global model_map
    global metadata_map
    model = model_map.get(language, None)
    metadata = metadata_map.get(language, None)
    if model and metadata:
        return model, metadata
    model, metadata = whisperx.load_align_model(language_code=language, device=util.get_device_type())
    model_map[language] = model
    metadata_map[language] = metadata
    return model, metadata


def exec_gc():
    global model_map
    global metadata_map
    model_map = None
    metadata_map = None
    util.exec_gc()


def transcribe(audio, segments, language):
    model, metadata = get_model(language)
    segments = tool_align_whisperx.transcribe(model, metadata, audio, segments)
    return segments, language
