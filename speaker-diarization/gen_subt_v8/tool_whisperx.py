import math
import numpy as np
import tool_subt


def pydub_whisperx(audio):
    # 转 numpy 数组（float32，范围 -1.0 ~ 1.0）
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples


def transcribe(model, audio, language=None):
    last_end = len(audio)

    samples = pydub_whisperx(audio)
    result = model.transcribe(samples, language=language, batch_size=16)

    segments = []
    for segment in result['segments']:
        start = math.floor(segment['start']* 1000)
        if start < 0:
            start = 0
        end = math.ceil(segment['end'] * 1000)
        if last_end < end:
            end = last_end
        segments.append({"start": start, "end": end, "text": segment['text']})

    if not language:
        language = result["language"]
    for i, segment in enumerate(segments):
        segments[i]['language'] = language

    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.clipp_segments(segments, last_end)
    segments = tool_subt.init_segments(segments)
    tool_subt.check_discrete_segments(segments)

    return segments, language
