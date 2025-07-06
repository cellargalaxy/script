from faster_whisper import WhisperModel
import util
import util_subt
from pydub import AudioSegment

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

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

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

    for i, segment in enumerate(subt['segments']):
        start = round(segment['start'] * 1000)
        if last_end <= start:
            continue
        end = round(segment['end'] * 1000)
        if last_end <= end:
            segment['end'] = last_end / 1000.0
        if segment['end'] < segment['start']:
            continue
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
    del segments
    util.exec_gc()

    return subt
