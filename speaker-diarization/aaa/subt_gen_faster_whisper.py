from faster_whisper import WhisperModel
import util

logger = util.get_logger()


def subt_gen(audio_path, auth_token):
    logger.info("字幕生成: %s", audio_path)

    model = WhisperModel("large-v3", device=util.get_device_type(), compute_type=util.get_compute_type())
    segments, info = model.transcribe(audio_path, beam_size=5)
    subt = {
        "segments": [],
        "word_segments": [],
        "language": info.language,
    }
    for segment in segments:
        obj = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        subt['segments'].append(obj)

    del model
    del segments
    util.exec_gc()

    return subt
