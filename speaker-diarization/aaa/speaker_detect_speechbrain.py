import util
import math
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
import util_subt
import torch
from speechbrain.inference.diarization import SpeakerDiarization

logger = util.get_logger()


def speaker_detect(audio_path, auth_token, min_speech_duration_ms=250):
    logger.info("说话人检测: %s", audio_path)

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    diarization_model = SpeakerDiarization.from_huggingface("speechbrain/spkrec-ecapa-voxceleb",
                                                            saved_dir="model/spkrec-ecapa-voxceleb")
    diarization_result = diarization_model.diarize_file(audio_path)


    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = math.floor(turn.start * 1000)
        if start < 0:
            start = 0
        end = math.ceil(turn.end * 1000)
        if last_end < end:
            end = last_end
        if end - start < min_speech_duration_ms:
            continue
        segments.append({"start": start, "end": end, "speaker": speaker})

    segments = util_subt.fix_overlap_segments(segments)

    del audio
    del pipeline
    del diarization
    util.exec_gc()

    return segments
