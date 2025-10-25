import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import util

logger = util.get_logger()


def loudness_normalization(input_path, output_path, target_lufs=-23.0):
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
    peak = np.max(np.abs(loudness_normalized_audio))

    logger.info(f"响度归一化,{input_path},原始响度:{loudness:.2f},目标响度:{target_lufs:.2f},归一化后峰值:{peak:.4f}")

    # 如果峰值超过1.0，说明可能会发生削波。
    # 在这种情况下，可以选择进行峰值归一化，将最大峰值压缩到1.0。
    # 这会稍微改变整体响度，但在防止失真方面是必要的。
    if peak > 1.0:
        loudness_normalized_audio = loudness_normalized_audio / peak
        # 重新检查峰值
        new_peak = np.max(np.abs(loudness_normalized_audio))
        logger.warn(f"响度归一化,{input_path},削波峰值:{new_peak:.4f}")

    # 5. 写入新文件
    # 使用sf.write写入文件。它会自动处理从float到原始整数格式的转换。
    # 通过从原始文件中读取的 'subtype' 信息，可以保留原始位深度。
    util.mkdir(output_path)
    info = sf.info(input_path)
    sf.write(output_path, loudness_normalized_audio, rate, subtype=info.subtype)


def get_loudness(audio, frame_rate=50):
    frame_simple = int(1000 / frame_rate)
    total_ms = len(audio)
    frame_cnt = total_ms // frame_simple
    volumes = []
    for i in range(frame_cnt):
        start_ms = i * frame_simple
        end_ms = (i + 1) * frame_simple
        frame_data = audio[start_ms: end_ms]
        dbfs = frame_data.dBFS
        volumes.append(dbfs)
    return volumes
