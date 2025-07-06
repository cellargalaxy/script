from faster_whisper import WhisperModel
import util

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


def segment_detect(audio_path, auth_token):
    logger.info("字幕生成: %s", audio_path)

    model = get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)
    subt = {
        "segments": [],
        "language": info.language,
    }
    for segment in segments:
        obj = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        subt['segments'].append(obj)

    return subt
