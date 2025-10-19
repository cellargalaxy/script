from pydub import AudioSegment
import util
import tool_ten_vad
import math
import tool_subt

logger = util.get_logger()


def diffusion_left(tags, tag):
    con = True
    while con:
        con = False
        for i, _ in enumerate(tags):
            if i == 0:
                continue
            if tags[i - 1] == 0 and tags[i] == tag:
                tags[i - 1] = tag
                con = True
                break
    return tags


def diffusion_right(tags, tag):
    con = True
    while con:
        con = False
        for i, _ in enumerate(tags):
            if i == 0:
                continue
            if tags[i - 1] == tag and tags[i] == 0:
                tags[i] = tag
                con = True
                break
    return tags


def part_detect(audio_path,
                frame_rate=50,
                speech_threshold=0.8,
                silence_threshold=0.2,
                ):
    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    confidences = tool_ten_vad.vad_confidence(audio, frame_rate)
    if len(confidences) == 0:
        logger.error(f"人声执行度为空")
        raise ValueError(f"人声执行度为空")

    tags = []
    for i, confidence in enumerate(confidences):
        if confidence <= silence_threshold:
            tags.append(-1)
        elif speech_threshold <= confidence:
            tags.append(1)
        else:
            tags.append(0)
    tags = diffusion_left(tags, 1)
    tags = diffusion_right(tags, 1)
    tags = diffusion_left(tags, -1)
    tags = diffusion_right(tags, -1)

    segments = []
    for i, tag in enumerate(tags):
        if tag == 1:
            vad_type = 'speech'
        elif tag == -1:
            vad_type = 'silence'
        else:
            logger.error(f"非法人声标签: {tag}")
            raise ValueError(f"非法人声标签: {tag}")
        if i > 0 and tags[i - 1] != tags[i]:
            pre_end = segments[-1]['end']
            segments.append({'start': pre_end, 'end': 0, 'vad_type': vad_type})
        if len(segments) == 0:
            segments.append({'start': 0, 'end': 0, 'vad_type': vad_type})
        end = math.floor((i + 1) * (1000.0 / frame_rate))
        segments[-1]['end'] = end

    if last_end < segments[-1]['end']:
        segments[-1]['end'] = last_end

    for i, segment in enumerate(segments):
        segments[i]['duration_ms'] = segments[i]['end'] - segments[i]['start']

    tool_subt.check_coherent_segments(segments)

    return segments
