from pydub import AudioSegment
import util
import tool_ten_vad
import math
import tool_subt
import tool_loudness

logger = util.get_logger()


def diffusion(tags, tag):
    for i, _ in enumerate(tags):
        if i == 0:
            continue
        if tags[i - 1] == tag and tags[i] == 0:
            tags[i] = tag
    for i in range(len(tags) - 2, -1, -1):
        if tags[i] == 0 and tags[i + 1] == tag:
            tags[i] = tag
    return tags


def part_detect_by_data(audio,
                        frame_rate=50,
                        speech_threshold=0.8,
                        silence_threshold=0.2,
                        volume_threshold=-80,
                        ):
    if audio.channels != 1:
        audio = audio.set_channels(1)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    last_end = len(audio)

    vads = tool_ten_vad.vad_confidence(audio, frame_rate)
    if len(vads) == 0:
        logger.error(f"人声置信度为空")
        raise ValueError(f"人声置信度为空")
    volumes = tool_loudness.get_loudness(audio, frame_rate)
    if len(volumes) == 0:
        logger.error(f"响度为空")
        raise ValueError(f"响度为空")
    if len(vads) != len(volumes):
        logger.error(f"人声置信度与响度长度不一致, vads: {len(vads)}, volumes: {len(volumes)}")
        raise ValueError(f"人声置信度与响度长度不一致, vads: {len(vads)}, volumes: {len(volumes)}")

    tags = []
    for i, vad in enumerate(vads):
        if vad <= silence_threshold:
            tags.append(-1)
        elif speech_threshold <= vad:
            tags.append(1)
        else:
            tags.append(0)

    for i, tag in enumerate(tags):
        if volumes[i] <= volume_threshold:
            tags[i] = -1

    tags = diffusion(tags, 1)
    tags = diffusion(tags, -1)
    for i, tag in enumerate(tags):
        if tags[i] == 0:
            tags[i] = 1

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
    if segments[-1]['end'] < last_end:
        pre_end = segments[-1]['end']
        segments.append({'start': pre_end, 'end': last_end, 'vad_type': 'silence'})

    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.unit_segments(segments, 'vad_type')
    segments = tool_subt.init_segments(segments)
    tool_subt.check_coherent_segments(segments)

    return segments


def part_detect(audio_path,
                frame_rate=50,
                speech_threshold=0.8,
                silence_threshold=0.2,
                volume_threshold=-80,
                ):
    audio = AudioSegment.from_wav(audio_path)
    segments = part_detect_by_data(audio,
                                   frame_rate=frame_rate,
                                   speech_threshold=speech_threshold,
                                   silence_threshold=silence_threshold,
                                   volume_threshold=volume_threshold,
                                   )
    return segments
