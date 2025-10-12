import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import util

logger = util.get_logger()


def loudness_match_one(input_path, output_path, target_lufs=-23.0):
    """
    对单个WAV文件进行响度归一化，并输出到新文件。

    参数:
    - input_path (str): 输入WAV文件的路径。
    - output_path (str): 输出WAV文件的路径。
    - target_lufs (float): 目标积分响度值 (LUFS)。
    """
    # 1. 读取音频文件
    # 使用 soundfile 读取，它会自动处理不同的位深度
    data, rate = sf.read(input_path)

    # 确保音频数据是浮点数类型，这是pyloudnorm和后续计算所必需的
    if not np.issubdtype(data.dtype, np.floating):
        # 找出原始整数类型的最大值
        max_val = np.iinfo(data.dtype).max
        # 转换为 [-1.0, 1.0] 范围内的浮点数
        data = data.astype(np.float32) / max_val

    # 如果是单声道，确保其shape为 (n_samples, 1) 以便处理
    if data.ndim == 1:
        data = data[:, np.newaxis]

    # 2. 测量原始响度
    # 创建响度计
    meter = pyln.Meter(rate)
    # 测量积分响度
    loudness = meter.integrated_loudness(data)

    # 打印原始响度信息

    # 3. 计算并应用增益
    # 根据 ITU-R BS.1770-4 标准进行响度归一化
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, target_lufs)

    # 4. 检查峰值，防止削波
    # 获取应用增益后的峰值
    peak_amplitude = np.max(np.abs(loudness_normalized_audio))

    logger.info(f"{input_path},原始响度:{loudness:.2f},目标响度:{target_lufs:.2f},归一化后峰值:{peak_amplitude:.4f}")

    # 如果峰值超过1.0，说明可能会发生削波。
    # 在这种情况下，可以选择进行峰值归一化，将最大峰值压缩到1.0。
    # 这会稍微改变整体响度，但在防止失真方面是必要的。
    if peak_amplitude > 1.0:
        loudness_normalized_audio = loudness_normalized_audio / peak_amplitude
        # 重新检查峰值
        new_peak = np.max(np.abs(loudness_normalized_audio))
        logger.warn(f"{input_path},削波峰值:{new_peak:.4f}")

    # 5. 写入新文件
    # 使用sf.write写入文件。它会自动处理从float到原始整数格式的转换。
    # 通过从原始文件中读取的 'subtype' 信息，可以保留原始位深度。
    info = sf.info(input_path)
    sf.write(output_path, loudness_normalized_audio, rate, subtype=info.subtype)


def loudness_match(speaker_path, output_dir):
    json_path = os.path.join(output_dir, 'loudness_match.json')
    if util.path_exist(json_path):
        return json_path

    speaks = util.read_file_to_obj(speaker_path)
    for i, speak in enumerate(speaks):
        segments = speaks[i]['segments']
        for j, segment in enumerate(segments):
            input_path = segments[j]['wav_path']
            output_path = os.path.join(output_dir, 'speaker', speaks[i]['file_name'], f"{segments[j]['file_name']}.wav")
            segments[j]['wav_path'] = output_path
            util.mkdir(output_path)
            loudness_match_one(input_path, output_path)
        speaks[i]['segments'] = segments

    util.save_as_json(speaks, json_path)
    return json_path


def exec(manager):
    logger.info("loudness_match,enter: %s", util.json_dumps(manager))
    speaker_export_path = manager.get('speaker_export_path')
    output_dir = os.path.join(manager.get('output_dir'), "loudness_match")
    loudness_match_path = loudness_match(speaker_export_path, output_dir)
    manager['loudness_match_path'] = loudness_match_path
    logger.info("loudness_match,leave: %s", util.json_dumps(manager))
    util.exec_gc()
