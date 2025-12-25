import math
import numpy as np
import tool_subt
from faster_whisper import WhisperModel


def pydub_faster_whisper(audio):
    # 转 numpy 数组（float32，范围 -1.0 ~ 1.0）
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples


def transcribe(model: WhisperModel, audio, language=None):
    last_end = len(audio)

    samples = pydub_faster_whisper(audio)
    # https://grok.com/c/78cde323-415f-4236-a193-79e630fcfc6e?rid=06a420c6-3037-47cf-9e2b-3d49bd49bca4
    results, info = model.transcribe(samples, language=language,
                                     vad_filter=True,
                                     vad_parameters=dict(min_silence_duration_ms=100),
                                     condition_on_previous_text=False,
                                     length_penalty=1.5,
                                     max_new_tokens=150,
                                     )

    segments = []
    for result in results:
        start = math.floor(result.start * 1000)
        if start < 0:
            start = 0
        end = math.ceil(result.end * 1000)
        if last_end < end:
            end = last_end
        segments.append({"start": start, "end": end, "text": result.text})

    if not language:
        language = info.language
    for i, segment in enumerate(segments):
        segments[i]['language'] = language

    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.clipp_segments(segments, last_end)
    segments = tool_subt.init_segments(segments)
    tool_subt.check_discrete_segments(segments)

    return segments, language
