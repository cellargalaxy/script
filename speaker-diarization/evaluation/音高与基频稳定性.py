"""
1. 我打算使用以下指标对ai翻唱的wav文件进行质量评价。
2. 我只有使用ai翻唱出来的多个wav文件，我能提供这些文件的路径。
3. 判断以下指标，只有wav文件路径，这些文件之间是否能对比出优劣，如果对比不出优劣就不需要再继续了
4. 如果能对比出优劣，写一个python函数，入参是wav文件路径的字符串数组
5. 该python函数实现以下指标的计算，并且将计算结果画为图表进行可视化对比
6. 图表的类型，需要根据指标的特点进行选择，目的是能更加直观的看出各个wav文件的优劣
7. 图表的数轴标度，为了避免不同文件之间的指标差异过小，在图中看不出区别，需要更加明显的处理
8. 文件大约有几十到一百个，需要合理排版，以能清晰看出每个文件的数据走向与图标
9. 文件路径数组已经排好序，按模型的轮数是递增的
10. 在图表中增加该指标的中文文字描述，阈值等辅助信息，使用文件名称标示出各个文件之间的差异
11. 文字描述不能遮住图表本身,将文字描述的背景颜色设置为透明，图表使用常规字体而不是等宽字体
12. 尽量将代码都收敛到函数内部，方便调用，按文件进行并发处理，提升处理速度
13. 最后提供一个完整可用的python函数，以及其需要安装的依赖

+ 音高与基频稳定性（Pitch / F0）
    + 含义：F0曲线是否平滑连续，无频繁断裂/跳变；使用PYIN或CREPE算法提取，评估模型/声码器稳定性（如锯齿状或死点表示AI
      artifact）。
    + 平滑、少自然抖动：正常歌声
    + 锯齿/断层明显：AI抖动或不稳
    + 在长音处应有自然颤音（Vibrato），转音处平滑过渡。
"""

# pip install numpy scipy pandas librosa soundfile matplotlib seaborn

from typing import List, Dict, Any
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import median_filter


def evaluate_pitch_f0_stability(
    wav_paths: List[str],
    sr: int = 22050,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 1100.0,
    top_db: float = 40.0,
    save_dir: str = None,
    title_prefix: str = "AI翻唱 Pitch/F0稳定性对比",
) -> pd.DataFrame:
    """
    基于单独wav文件（无参考真值）对“音高/F0稳定性”做相对质量评价与可视化对比。
    适用：同一首歌/相近内容，按训练轮数递增的多个生成结果。

    入参:
        wav_paths: wav文件路径列表（已排序：轮数递增）
        sr: 重采样采样率
        hop_length: F0提取帧移
        fmin/fmax: pyin搜索范围
        top_db: 去静音阈值（越小保留越多低能量段）
        save_dir: 保存图表的目录；None则不保存，只显示
        title_prefix: 图表标题前缀

    返回:
        DataFrame: 每个文件的各项F0稳定性指标
    """

    # ----------------------------
    # 0) 基本检查与风格设置
    # ----------------------------
    if not wav_paths:
        raise ValueError("wav_paths 不能为空")

    for p in wav_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"文件不存在: {p}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial Unicode MS", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_style("whitegrid")

    # ----------------------------
    # 1) 指标定义（经验阈值/解释用）
    # ----------------------------
    # 注：这些阈值是“经验参考”，不是通用标准。主要用于图上辅助判断。
    thresholds = {
        "voiced_ratio": (0.45, 0.95),          # 发声占比过低可能F0提取困难/静音多；过高也未必坏
        "f0_break_rate": (0.00, 0.08),         # voiced内部的断裂率，越低越好
        "octave_jump_rate": (0.00, 0.05),      # 近似“倍频/半频”跳变率，越低越好
        "semitone_jitter_mad": (0.00, 0.25),   # 去趋势后半音抖动MAD，越低越稳（自然颤音会抬高）
        "f0_smoothness_med": (0.00, 0.35),     # 一阶差分(半音)中位数，越低越平滑
        "vibrato_presence": (0.10, 0.60),      # 长音中4-8Hz能量占比，适中更像自然颤音（过高/过低都可能异常）
    }

    # ----------------------------
    # 2) 逐文件提取F0 + 计算指标
    # ----------------------------
    rows = []

    def _safe_base_name(path: str, max_len: int = 30) -> str:
        b = os.path.splitext(os.path.basename(path))[0]
        return b if len(b) <= max_len else (b[: max_len - 3] + "...")

    for idx, path in enumerate(wav_paths):
        y, _sr = librosa.load(path, sr=sr, mono=True)

        # 去静音（减少静音段对voiced_ratio等的影响）
        yt, _ = librosa.effects.trim(y, top_db=top_db)

        # 防止极短音频
        if len(yt) < sr * 1.0:
            yt = y  # 回退，不trim

        # pyin 提取
        f0, voiced_flag, voiced_prob = librosa.pyin(
            yt,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
        )

        f0 = np.array(f0, dtype=float)  # NaN 表示 unvoiced
        voiced_flag = np.array(voiced_flag, dtype=bool)

        n = len(f0)
        if n < 5:
            # 过短直接给NaN指标
            rows.append({
                "index": idx,
                "file": _safe_base_name(path),
                "path": path,
                "voiced_ratio": np.nan,
                "f0_median_hz": np.nan,
                "f0_iqr_hz": np.nan,
                "f0_break_rate": np.nan,
                "octave_jump_rate": np.nan,
                "f0_smoothness_med": np.nan,
                "semitone_jitter_mad": np.nan,
                "vibrato_presence": np.nan,
            })
            continue

        # voiced ratio
        voiced_ratio = np.nanmean(voiced_flag.astype(float))

        # 只在voiced部分分析连续性/跳变
        f0_voiced = f0.copy()
        # voiced内部断裂：voiced_flag从True到False算断裂；但我们仅统计“在voiced段中出现缺失”的情况
        # 简化：统计 NaN 的比例（在trim后） + voiced段边界变化频率
        nan_ratio = np.mean(np.isnan(f0_voiced))
        # voiced内部断裂率：voiced_flag变化次数 / 总帧数
        flag_changes = np.sum(voiced_flag[1:] != voiced_flag[:-1])
        f0_break_rate = flag_changes / max(1, (n - 1))

        # 计算半音序列（对voiced部分）
        # 只对有效f0计算，避免NaN
        valid = ~np.isnan(f0_voiced)
        f0_valid = f0_voiced[valid]
        if len(f0_valid) < 10:
            f0_median = np.nan
            f0_iqr = np.nan
            octave_jump_rate = np.nan
            f0_smoothness_med = np.nan
            semitone_jitter_mad = np.nan
            vibrato_presence = np.nan
        else:
            f0_median = float(np.median(f0_valid))
            f0_iqr = float(np.subtract(*np.percentile(f0_valid, [75, 25])))

            # 半音转换: st = 12*log2(f0/440)
            st = 12.0 * np.log2(f0_valid / 440.0)

            # 平滑性：一阶差分(半音)的中位数绝对值
            dst = np.diff(st)
            f0_smoothness_med = float(np.median(np.abs(dst)))

            # 倍频/半频跳变（近似八度跳）：相邻差分绝对值 > 10 半音（可调整）
            octave_jump_rate = float(np.mean(np.abs(dst) > 10.0))

            # 抖动/不稳：去掉慢变化趋势，再看残差的MAD
            # 用中值滤波提取趋势（窗口约 ~200ms）
            win = int(round(0.2 * sr / hop_length))
            win = max(3, win if win % 2 == 1 else win + 1)
            trend = median_filter(st, size=win, mode="nearest")
            resid = st - trend
            semitone_jitter_mad = float(np.median(np.abs(resid - np.median(resid))))

            # 颤音存在度：在残差中4-8Hz能量占比（粗略）
            # 将resid视为等间隔采样，采样率为 frame_rate
            frame_rate = sr / hop_length
            r = resid - np.mean(resid)
            # FFT
            fft = np.fft.rfft(r)
            freqs = np.fft.rfftfreq(len(r), d=1.0 / frame_rate)
            psd = (np.abs(fft) ** 2)

            band = (freqs >= 4.0) & (freqs <= 8.0)
            total = (freqs >= 0.5) & (freqs <= 12.0)
            vibrato_presence = float(psd[band].sum() / (psd[total].sum() + 1e-12))

        rows.append({
            "index": idx,
            "file": _safe_base_name(path),
            "path": path,
            "voiced_ratio": float(voiced_ratio),
            "f0_median_hz": float(f0_median) if np.isfinite(f0_median) else np.nan,
            "f0_iqr_hz": float(f0_iqr) if np.isfinite(f0_iqr) else np.nan,
            "f0_break_rate": float(f0_break_rate),  # voiced/unvoiced切换频繁 -> 断裂/不稳
            "octave_jump_rate": float(octave_jump_rate) if np.isfinite(octave_jump_rate) else np.nan,
            "f0_smoothness_med": float(f0_smoothness_med) if np.isfinite(f0_smoothness_med) else np.nan,
            "semitone_jitter_mad": float(semitone_jitter_mad) if np.isfinite(semitone_jitter_mad) else np.nan,
            "vibrato_presence": float(vibrato_presence) if np.isfinite(vibrato_presence) else np.nan,
            "nan_ratio": float(nan_ratio),
        })

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)

    # ----------------------------
    # 3) 可视化：多文件适配（热力图 + 趋势折线）
    # ----------------------------
    # 指标选择（越低越好/越高越好混合）
    metrics = [
        "voiced_ratio",
        "f0_break_rate",
        "octave_jump_rate",
        "f0_smoothness_med",
        "semitone_jitter_mad",
        "vibrato_presence",
    ]

    # 3.1 robust缩放：按(5%,95%)拉伸到[0,1]，增强“差异过小”的可见性
    scaled = {}
    for m in metrics:
        x = df[m].to_numpy(dtype=float)
        lo = np.nanpercentile(x, 5) if np.isfinite(np.nanmedian(x)) else np.nan
        hi = np.nanpercentile(x, 95) if np.isfinite(np.nanmedian(x)) else np.nan
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            scaled[m] = np.full_like(x, np.nan, dtype=float)
        else:
            z = (x - lo) / (hi - lo)
            scaled[m] = np.clip(z, 0.0, 1.0)

    df_scaled = pd.DataFrame(scaled)
    df_scaled["file"] = df["file"]

    # 3.2 z-score（每个指标内标准化）用于热力图更直观显示“相对好坏”
    zmap = {}
    for m in metrics:
        x = df[m].to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x) + 1e-12
        zmap[m] = (x - mu) / sd
    df_z = pd.DataFrame(zmap)
    df_z["file"] = df["file"]

    # 图表尺寸：几十~100个文件时，热力图高度要拉大
    nfiles = len(df)
    heat_h = max(6, min(24, 0.22 * nfiles))  # 控制最大高度，避免过大
    line_h = 7

    # --- A) 热力图：z-score（相对表现）
    fig1, ax1 = plt.subplots(figsize=(14, heat_h), constrained_layout=True)
    hm_data = df_z.set_index("file")[metrics]
    sns.heatmap(
        hm_data,
        ax=ax1,
        cmap="RdBu_r",
        center=0.0,
        cbar_kws={"label": "z-score（相对同批次均值）"},
        linewidths=0.2,
        linecolor=(0, 0, 0, 0.08),
    )
    ax1.set_title(f"{title_prefix} - 指标热力图（z-score）", fontsize=14)
    ax1.set_xlabel("指标")
    ax1.set_ylabel("文件（按轮数递增）")

    # 透明文字说明（不遮挡主体：放到图外右侧/下方用fig.text）
    desc = (
        "指标说明（相对对比用，无参考真值）：\n"
        "- voiced_ratio：有声音高帧占比（过低可能难提F0/静音多）\n"
        "- f0_break_rate：voiced/unvoiced切换频繁度（越低越连贯）\n"
        "- octave_jump_rate：近似八度跳变率（倍频/半频错误，越低越好）\n"
        "- f0_smoothness_med：半音一阶差分中位数（越低越平滑）\n"
        "- semitone_jitter_mad：去趋势后的半音残差MAD（越低越稳；颤音会抬高）\n"
        "- vibrato_presence：4-8Hz能量占比（适中更像自然颤音）"
    )
    fig1.text(
        0.01, 0.01, desc,
        ha="left", va="bottom",
        fontsize=10,
        bbox=dict(facecolor=(1, 1, 1, 0.0), edgecolor=(0, 0, 0, 0.0), pad=6)
    )

    if save_dir:
        fig1.savefig(os.path.join(save_dir, "f0_metrics_heatmap_zscore.png"), dpi=200)

    # --- B) 趋势折线：对“越低越好”的指标画原始值

