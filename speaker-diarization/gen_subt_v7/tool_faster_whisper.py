import math
import util
import numpy as np
import tool_subt

logger = util.get_logger()


def pydub_faster_whisper(audio):
    # 转 numpy 数组（float32，范围 -1.0 ~ 1.0）
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples


def transcribe(model, audio):
    last_end = len(audio)

    samples = pydub_faster_whisper(audio)
    results, info = model.transcribe(samples)

    segments = []
    for result in results:
        start = math.floor(result.start * 1000)
        if start < 0:
            start = 0
        end = math.ceil(result.end * 1000)
        if last_end < end:
            end = last_end
        segments.append({"start": start, "end": end, "text": result.text})

    segments = tool_subt.init_segments(segments)
    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.clipp_segments(segments, last_end)
    tool_subt.check_discrete_segments(segments)

    return segments
