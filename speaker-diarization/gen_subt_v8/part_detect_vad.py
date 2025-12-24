from pydub import AudioSegment
import util
import tool_ten_vad
import math
import tool_subt
import tool_loudness
import tool_nemo_vad

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
                        vad_speech_threshold=0.8,
                        vad_silence_threshold=0.2,
                        volume_speech_threshold=-30,
                        volume_silence_threshold=-70,
                        ):
    last_end = len(audio)

    volumes = tool_loudness.get_loudness(audio, frame_rate)
    if len(volumes) == 0:
        logger.error(f"响度为空")
        raise ValueError(f"响度为空")
    ten_vads = tool_ten_vad.vad_confidence(audio, frame_rate)
    if len(ten_vads) == 0:
        logger.error(f"人声置信度为空")
        raise ValueError(f"人声置信度为空")
    nemo_vads = tool_nemo_vad.vad_confidence(audio)
    if len(nemo_vads) == 0:
        logger.error(f"人声置信度为空")
        raise ValueError(f"人声置信度为空")
    if len(volumes) - len(nemo_vads) == 1:
        nemo_vads.append(0)
    if len(nemo_vads) - len(volumes) == 1:
        nemo_vads.pop()
    if not len(volumes) == len(ten_vads) == len(nemo_vads):
        logger.error(
            f"人声置信度与响度长度不一致, volumes: {len(volumes)}, ten_vads: {len(ten_vads)}, nemo_vads: {len(nemo_vads)}")
        raise ValueError(
            f"人声置信度与响度长度不一致, volumes: {len(volumes)}, ten_vads: {len(ten_vads)}, nemo_vads: {len(nemo_vads)}")

    plot_volume(nemo_vads)
    tags = [0] * len(ten_vads)
    # for i, volume in enumerate(volumes):
    #     if tags[i] == 0 and volume_speech_threshold <= volumes[i]:
    #         tags[i] = 1
    #     if tags[i] == 0 and volumes[i] <= volume_silence_threshold:
    #         tags[i] = -1
    # for i, vad in enumerate(ten_vads):
    #     if tags[i] == 0 and vad <= vad_silence_threshold:
    #         tags[i] = -1
    #     elif tags[i] == 0 and vad_speech_threshold <= vad:
    #         tags[i] = 1
    for i, vad in enumerate(nemo_vads):
        if tags[i] == 0 and vad <= vad_silence_threshold:
            tags[i] = -1
        elif tags[i] == 0 and vad_speech_threshold <= vad:
            tags[i] = 1

    tags = diffusion(tags, 1)
    tags = diffusion(tags, -1)

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


import math
import matplotlib.pyplot as plt


def plot_volume(volume_array):
    """
    volume_array: List[float]
        每个元素代表 20ms 内的声量，可能包含 -Infinity
    """

    # 处理 -Infinity（用 None 或一个很小的值）
    cleaned_volume = []
    for v in volume_array:
        if v == -math.inf:
            cleaned_volume.append(None)  # matplotlib 会断开曲线
        else:
            cleaned_volume.append(v)

    # 生成时间轴（单位：秒）
    time_axis = [i * 0.02 for i in range(len(cleaned_volume))]

    # 画图
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, cleaned_volume)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Volume")
    plt.title("Audio Volume Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def part_detect(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    segments = part_detect_by_data(audio)
    return segments
