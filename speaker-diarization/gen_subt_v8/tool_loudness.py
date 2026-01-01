import numpy as np
import pyloudnorm as pyln
import util
from pydub import AudioSegment

logger = util.get_logger()


def normalize_loudness(
        audio: AudioSegment,
        target_lufs: float = -18.0,
        peak_ceiling_db: float = -1.0,
        silence_threshold_lufs: float = -70.0,
) -> AudioSegment:
    """
    针对AI语音模型训练优化的ITU-R BS.1770-4响度归一化。

    设计原则：
        1. 高精度：全程float64处理
        2. 无削波：峰值限制优先于响度目标
        3. 稳健性：静音/短音频优雅降级
        4. 位深保持：输出与输入位深一致

    Args:
        audio: pydub AudioSegment对象
        target_lufs: 目标集成响度(LUFS)
            - -23.0: EBU R128广播标准
            - -20.0: 语音训练常用值（推荐）
            - -16.0: 高响度，适合清晰语音
        peak_ceiling_db: 峰值上限(dBFS)，默认-1.0dB
            防止重采样/编码时intersample peak削波
        silence_threshold_lufs: 静音判定阈值，低于此值不处理
    """

    # ======================= 时长检查 =======================
    # ITU-R BS.1770门控块最小400ms，pyloudnorm需要足够样本
    MIN_DURATION_MS = 400
    if len(audio) < MIN_DURATION_MS:
        return audio

    # ======================= 提取音频参数 =======================
    sample_rate = audio.frame_rate
    channels = audio.channels
    original_sample_width = audio.sample_width

    # ======================= 转换为float64数组 =======================
    # 策略：统一转为32-bit int再转float64，确保各位深兼容性
    if original_sample_width != 4:
        work_audio = audio.set_sample_width(4)
    else:
        work_audio = audio

    samples = np.array(work_audio.get_array_of_samples(), dtype=np.float64)
    samples /= 2147483648.0  # 2^31，归一化到 [-1.0, 1.0)

    # pyloudnorm要求多声道shape为 (n_samples, n_channels)
    if channels > 1:
        samples = samples.reshape((-1, channels))

    # ======================= 空音频检查 =======================
    if np.max(np.abs(samples)) < 1e-10:
        return audio

    # ======================= LUFS测量 =======================
    meter = pyln.Meter(sample_rate)

    try:
        current_lufs = meter.integrated_loudness(samples)
    except Exception as e:
        logger.error("响度归一化，异常: %s", e)
        return audio

    # 静音/极低响度检查
    if not np.isfinite(current_lufs) or current_lufs < silence_threshold_lufs:
        return audio

    # ======================= 响度归一化 =======================
    # 使用pyloudnorm标准归一化（浮点域精确处理）
    normalized_samples = pyln.normalize.loudness(samples, current_lufs, target_lufs)

    # ======================= 峰值限制 =======================
    # 防止削波：对AI训练至关重要（削波引入高频谐波噪声）
    peak_ceiling_linear = 10.0 ** (peak_ceiling_db / 20.0)
    current_peak = np.max(np.abs(normalized_samples))

    if current_peak > peak_ceiling_linear:
        # 线性缩放限制峰值
        scale = peak_ceiling_linear / current_peak
        normalized_samples *= scale

    # ======================= 转换回整数格式 =======================
    # 保持原始位深度
    out_width = original_sample_width

    # 位深度配置: (最大值, numpy类型)
    # 注意：有符号整数范围是 [-max_val, max_val-1]
    DTYPE_MAP = {
        1: (128.0, np.int8),  # 8-bit: -128 ~ 127
        2: (32768.0, np.int16),  # 16-bit: -32768 ~ 32767
        4: (2147483648.0, np.int32)  # 32-bit
    }

    if out_width not in DTYPE_MAP:
        out_width = 2  # fallback到16-bit

    max_val, dtype = DTYPE_MAP[out_width]

    # 缩放、裁剪、转换类型
    normalized_int = normalized_samples * max_val
    # clip确保不溢出（理论上经过峰值限制后不应超，但作为安全措施）
    normalized_int = np.clip(normalized_int, -max_val, max_val - 1)
    normalized_int = normalized_int.astype(dtype)

    # 多声道：恢复交错格式 (flatten)
    if channels > 1:
        normalized_int = normalized_int.flatten()

    # ======================= 构建输出AudioSegment =======================
    normalized_audio = AudioSegment(
        data=normalized_int.tobytes(),
        sample_width=out_width,
        frame_rate=sample_rate,
        channels=channels
    )

    return normalized_audio


def get_loudness(audio: AudioSegment, frame_rate: int = 50):
    simple_rate = 1000
    frame_simple = int(simple_rate / frame_rate)
    window_ms = int(1000 / frame_rate)
    frame_cnt = len(audio) // frame_simple
    volumes = [0.0] * len(audio)
    for i in range(frame_cnt):
        sample_start = i * frame_simple
        sample_end = (i + 1) * frame_simple
        frame_data = audio[sample_start:sample_end]
        volume = frame_data.dBFS
        ms_start = i * window_ms
        ms_end = min((i + 1) * window_ms, len(audio))
        for ms in range(ms_start, ms_end):
            volumes[ms] = volume
    return volumes
