import whisper_timestamped as whisper
import util

logger = util.get_logger()


def subt_gen(audio_path, auth_token):
    logger.info("字幕生成: %s", audio_path)

    model = whisper.load_model("large-v3", device=util.get_device_type())
    audio = whisper.load_audio(audio_path)
    result = whisper.transcribe(model, audio)
    subt = {
        "segments": [],
        "word_segments": [],
        "language": result["language"],
    }
    segments = result['segments']
    for segment in segments:
        words = []
        for word in segment['words']:
            obj = {
                "word": word['text'],
                "start": word['start'],
                "end": word['end'],
                "score": word['confidence'],
            }
            words.append(obj)
            subt['word_segments'].append(obj)
        obj = {
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'],
            "words": words,
        }
        subt['segments'].append(obj)

    del model
    del audio
    util.exec_gc()

    return subt
