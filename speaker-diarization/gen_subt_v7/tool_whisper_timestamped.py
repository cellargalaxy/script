import whisper_timestamped as whisper
import numpy as np
import tool_subt
import math


def pydub_faster_whisper(audio):
    # 转 numpy 数组（float32，范围 -1.0 ~ 1.0）
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples


def transcribe(model, audio):
    last_end = len(audio)

    samples = pydub_faster_whisper(audio)
    result = whisper.transcribe(model, samples)

    segments = result['segments']
    results = []
    for segment in segments:
        start = math.floor(segment['start'] * 1000)
        if start < 0:
            start = 0
        end = math.ceil(segment['end'] * 1000)
        if last_end < end:
            end = last_end
        segment_dict = {'start': start, 'end': end, 'text': segment['text']}
        results.append(segment_dict)

    results = tool_subt.fix_overlap_segments(results)
    results = tool_subt.clipp_segments(results, last_end)
    results = tool_subt.init_segments(results)
    tool_subt.check_discrete_segments(results)

    return results
