import util
import whisperx
import numpy as np
import math
import tool_subt


def pydub_faster_whisper(audio):
    # 转 numpy 数组（float32，范围 -1.0 ~ 1.0）
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples


def transcribe(model, metadata, audio, segments):
    last_end = len(audio)
    samples = pydub_faster_whisper(audio)

    segments = util.deepcopy_obj(segments)
    for i, segment in enumerate(segments):
        segments[i]['start'] /= 1000.0
        segments[i]['end'] /= 1000.0

    device = util.get_device_type()
    align_result = whisperx.align(segments, model, metadata, samples, device, return_char_alignments=False)

    segments = []
    for i, result in enumerate(align_result['segments']):
        start = math.floor(result['start'] * 1000)
        if start < 0:
            start = 0
        end = math.ceil(result['end'] * 1000)
        if last_end < end:
            end = last_end
        segments.append({"start": start, "end": end, "text": result['text']})

    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.clipp_segments(segments, last_end)
    segments = tool_subt.init_segments(segments)
    tool_subt.check_discrete_segments(segments)

    return segments
