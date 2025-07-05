import math

import whisper_timestamped as whisper
import util
import util_subt
from pydub import AudioSegment

logger = util.get_logger()


def subt_gen(audio_path, auth_token):
    logger.info("字幕生成: %s", audio_path)

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    model = whisper.load_model("large-v3", device=util.get_device_type())
    audio = whisper.load_audio(audio_path)
    result = whisper.transcribe(model, audio)
    subt = {
        "segments": [],
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
        obj = {
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'],
            "words": words,
        }
        subt['segments'].append(obj)

    segments = []
    for i, segment in enumerate(subt['segments']):
        start = round(segment['start'] * 1000)
        if last_end <= start:
            continue
        end = round(segment['end'] * 1000)
        if last_end <= end:
            segment['end'] = math.floor(last_end / 1000.0)
        segments.append(segment)
    subt['segments'] = segments

    segments = []
    for i, segment in enumerate(subt['segments']):
        if i == 0:
            segments.append(subt['segments'][i])
            continue
        if subt['segments'][i]['start'] < subt['segments'][i - 1]['end']:
            continue
        segments.append(subt['segments'][i])
    subt['segments'] = segments

    util_subt.check_discrete_segments(subt['segments'])

    del model
    del audio
    util.exec_gc()

    return subt
