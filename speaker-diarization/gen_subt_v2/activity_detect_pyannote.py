import util_subt
from pydub import AudioSegment
from pyannote.audio import Pipeline
import math
import util
import torch
import util_vad

logger = util.get_logger()


def activity_detect(audio_path, auth_token):
    logger.info("活动检测: %s", audio_path)

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=auth_token)
    pipeline = pipeline.to(torch.device(util.get_device_type()))
    results = pipeline(audio_path)

    segments = []
    for result in results.get_timeline():
        pre_end = 0
        if len(segments) > 0:
            pre_end = segments[-1]['end']
        start = math.floor(result.start * 1000)
        if start < 0:
            start = 0
        end = math.ceil(result.end * 1000)
        if last_end < end:
            end = last_end
        if pre_end < start:
            segments.append({"start": pre_end, "end": start, "vad_type": 'silene'})
        if start < end:
            segments.append({"start": start, "end": end, "vad_type": 'speech'})
        cut = audio[start:end]
        has_speech, max_probability, probability_ms = util_vad.has_speech_by_data(cut)
        if not has_speech:
            segments[-1]['vad_type'] = 'silene'

    pre_end = 0
    if len(segments) > 0:
        pre_end = segments[-1]['end']
    if pre_end < last_end:
        segments.append({"start": pre_end, "end": last_end, "vad_type": 'silene'})

    segments = util_subt.fix_overlap_segments(segments)
    segments = util_subt.unit_segments(segments, 'vad_type')
    util_subt.check_coherent_segments(segments)

    del audio
    del pipeline
    del results
    util.exec_gc()

    return segments
