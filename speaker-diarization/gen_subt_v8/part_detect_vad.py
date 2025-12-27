from pydub import AudioSegment
import util
import tool_ten_vad
import tool_subt
import tool_loudness
import os

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


def part_detect(audio_path, output_dir,
                vad_speech_threshold=0.8,
                vad_silence_threshold=0.2,
                volume_silence_threshold=-70,
                ):
    volume_path = os.path.join(output_dir, 'volume.json')
    vad_path = os.path.join(output_dir, 'vad.json')
    tag_path = os.path.join(output_dir, 'tag.json')

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    volumes = tool_loudness.get_loudness(audio)
    util.save_as_json(volumes, volume_path)
    if len(volumes) == 0:
        logger.error(f"响度为空")
        raise ValueError(f"响度为空")
    ten_vads = tool_ten_vad.vad_confidence(audio)
    util.save_as_json(ten_vads, vad_path)
    if len(ten_vads) == 0:
        logger.error(f"人声置信度为空")
        raise ValueError(f"人声置信度为空")
    if not len(volumes) == len(ten_vads):
        logger.error(f"人声置信度与响度长度不一致, volumes:{len(volumes)}")
        logger.error(f"人声置信度与响度长度不一致, ten_vads:{len(ten_vads)}")
        raise ValueError(f"人声置信度与响度长度不一致")

    tags = [0] * last_end
    for i, vad in enumerate(ten_vads):
        if tags[i] == 0 and vad <= vad_silence_threshold:
            tags[i] = -1
        elif tags[i] == 0 and vad_speech_threshold <= vad:
            tags[i] = 1
    for i, volume in enumerate(volumes):
        if tags[i] == 0 and volumes[i] <= volume_silence_threshold:
            tags[i] = -1

    tags = diffusion(tags, 1)
    tags = diffusion(tags, -1)
    util.save_as_json(tags, tag_path)

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
        segments[-1]['end'] = i + 1

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
