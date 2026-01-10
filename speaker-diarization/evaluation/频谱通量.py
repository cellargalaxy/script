"""
1. 我打算使用以下指标对ai翻唱的wav文件进行质量评价。
2. 我只有使用ai翻唱出来的多个wav文件，我能提供这些文件的路径。
3. 判断以下指标，只有wav文件路径，这些文件之间是否能对比出优劣，如果对比不出优劣就不需要再继续了
4. 如果能对比出优劣，写一个python函数，入参是wav文件路径的字符串数组
5. 该python函数实现以下指标的计算，并且将计算结果画为图表进行可视化对比，弹出窗口展示该图表
6. 图表的类型，需要根据指标的特点进行选择，目的是能更加直观的看出各个wav文件的优劣
7. 图表的数轴标度，为了避免不同文件之间的指标差异过小，在图中看不出区别，需要更加明显的处理
8. 文件大约有几十到一百个，需要合理排版，以能清晰看出每个文件的数据走向与图标
9. 文件路径数组已经排好序，按模型的轮数是递增的
10. 在图表中增加该指标的中文文字描述，阈值等辅助信息，使用文件名称标示出各个文件之间的差异
11. 将文字描述的背景颜色设置为透明，图表使用常规字体而不是等宽字体
12. 尽量将代码都收敛到函数内部，方便调用，按文件进行并发处理，提升处理速度
13. 最后提供一个完整可用的python函数，以及其需要安装的依赖

+ 频谱通量（Spectral Flux）
    + 含义：相邻帧频谱变化幅度，公式：`Flux = √[Σ |A_t(f) - A_{t-1}(f)|^2]`；检测节奏一致性或AI跳变。
    + 0.01–0.1：平稳过渡
    + 0.1–0.5：有节奏
    + 大于0.5：频繁跳变
"""

# pip install numpy matplotlib librosa soundfile

"""
频谱通量分析工具
用于评估AI翻唱音频质量

依赖安装：
pip install numpy matplotlib librosa soundfile

"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
import warnings


def analyze_spectral_flux(wav_paths: List[str], max_workers: int = 8) -> dict:
    """
    分析多个WAV文件的频谱通量并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数递增排序）
        max_workers: 并发处理的最大线程数

    返回:
        dict: 包含各文件分析结果的字典
    """

    warnings.filterwarnings('ignore')

    # ==================== 字体设置（常规字体，非等宽）====================
    plt.rcParams.update({
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'PingFang SC',
                            'Hiragino Sans GB', 'WenQuanYi Micro Hei',
                            'Noto Sans CJK SC', 'DejaVu Sans', 'Arial'],
        'font.family': 'sans-serif',
        'axes.unicode_minus': False,
        'font.size': 10,
    })

    # ==================== 单文件处理函数 ====================
    def compute_single_file(args: Tuple[int, str]) -> Tuple[int, str, float, float, float, bool]:
        """
        计算单个文件的频谱通量

        返回: (索引, 文件名, 平均值, P5值, P95值, 是否成功)
        """
        idx, wav_path = args
        filename = Path(wav_path).stem

        try:
            # 加载音频（统一采样率确保可比性）
            y, sr = librosa.load(wav_path, sr=22050, mono=True)

            # 计算短时傅里叶变换的幅度谱
            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

            # L2归一化（每帧独立归一化，消除音量影响）
            frame_norms = np.linalg.norm(S, axis=0, keepdims=True) + 1e-10
            S_norm = S / frame_norms

            # 计算频谱通量: Flux = √[Σ |A_t(f) - A_{t-1}(f)|²]
            # 使用RMS使数值落在合理范围
            diff = np.diff(S_norm, axis=1)
            flux_per_frame = np.sqrt(np.mean(diff ** 2, axis=0))

            # 统计量（使用分位数避免极端值干扰）
            mean_flux = float(np.mean(flux_per_frame))
            p5_flux = float(np.percentile(flux_per_frame, 5))
            p95_flux = float(np.percentile(flux_per_frame, 95))

            return (idx, filename, mean_flux, p5_flux, p95_flux, True)

        except Exception as e:
            print(f"  ⚠ 处理失败 [{filename}]: {e}")
            return (idx, filename, np.nan, np.nan, np.nan, False)

    # ==================== 并发处理 ====================
    print(f"🎵 开始分析 {len(wav_paths)} 个WAV文件...")
    print("=" * 50)

    results_dict = {}
    tasks = [(i, path) for i, path in enumerate(wav_paths)]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(wav_paths))) as executor:
        futures = {executor.submit(compute_single_file, task): task[0] for task in tasks}

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results_dict[result[0]] = result
            completed += 1

            # 进度显示
            if completed % max(1, len(wav_paths) // 10) == 0 or completed == len(wav_paths):
                pct = completed / len(wav_paths) * 100
                print(f"  处理进度: {completed}/{len(wav_paths)} ({pct:.0f}%)")

    # 按原始顺序整理结果
    results = [results_dict[i] for i in range(len(wav_paths))]

    # 提取数据
    filenames = [r[1] for r in results]
    mean_fluxes = np.array([r[2] for r in results])
    p5_fluxes = np.array([r[3] for r in results])
    p95_fluxes = np.array([r[4] for r in results])
    valid_mask = np.array([r[5] for r in results])

    print("=" * 50)
    print("✅ 数据分析完成，正在生成可视化图表...")

    # ==================== 可视化 ====================
    n_files = len(wav_paths)
    fig_width = max(14, min(24, n_files * 0.2))
    fig = plt.figure(figsize=(fig_width, 13))

    x = np.arange(n_files)

    # 阈值定义
    THRESHOLDS = [
        (0.01, '平稳过渡下限', '#2E8B57', '--'),
        (0.1, '节奏变化阈值', '#FF8C00', '--'),
        (0.5, '频繁跳变阈值', '#DC143C', '--'),
    ]

    # ---------- 子图1: 趋势线图 ----------
    ax1 = fig.add_subplot(2, 1, 1)

    # 绘制波动范围（P5-P95）
    valid_x = x[valid_mask]
    ax1.fill_between(valid_x,
                     p5_fluxes[valid_mask],
                     p95_fluxes[valid_mask],
                     alpha=0.25, color='#4169E1',
                     label='波动范围 (P5-P95)')

    # 绘制均值曲线
    marker_size = max(3, min(8, 150 // n_files))
    line_width = max(1, min(2, 80 // n_files))
    ax1.plot(valid_x, mean_fluxes[valid_mask], '-o',
             color='#4169E1',
             markersize=marker_size,
             linewidth=line_width,
             markerfacecolor='white',
             markeredgewidth=1.5,
             label='平均频谱通量',
             zorder=5)

    # 动态Y轴范围（主要修改点：优化数据区间展示）
    valid_data = mean_fluxes[valid_mask]
    if len(valid_data) > 0:
        # 获取数据的实际范围
        data_min = np.min(p5_fluxes[valid_mask])
        data_max = np.max(p95_fluxes[valid_mask])
        data_range = data_max - data_min

        # 如果数据范围太小（小于最大值的5%），则扩展范围以显示差异
        if data_range < data_max * 0.05:
            data_min = data_min - data_max * 0.1  # 向下扩展10%
            data_max = data_max + data_max * 0.1  # 向上扩展10%
            data_range = data_max - data_min

        # 计算合适的Y轴边界，确保数据占主要空间
        y_padding = data_range * 0.15  # 15%的边距
        y_bottom = max(0, data_min - y_padding)
        y_top = data_max + y_padding

        # 确保Y轴范围有意义
        if y_top - y_bottom < data_max * 0.05:
            y_top = data_max + data_max * 0.1
            y_bottom = max(0, data_min - data_min * 0.1)

        # 设置Y轴范围
        ax1.set_ylim(y_bottom, y_top)

    # 检查阈值线是否在数据范围内，决定是否显示
    for thresh_val, thresh_label, color, style in THRESHOLDS:
        if len(valid_data) > 0:
            # 只在阈值线接近数据范围时才显示
            y_bottom, y_top = ax1.get_ylim()
            if y_bottom <= thresh_val <= y_top:
                ax1.axhline(y=thresh_val, color=color, linestyle=style,
                           linewidth=1.5, alpha=0.6)
                # 将阈值标签放在图例中，避免占用空间
                ax1.plot([], [], color=color, linestyle=style, linewidth=1.5,
                        label=f'{thresh_label} ({thresh_val})', alpha=0.7)

    ax1.set_xlabel('文件序号（按模型训练轮数递增 →）', fontsize=11, fontweight='bold')
    ax1.set_ylabel('频谱通量', fontsize=11, fontweight='bold')
    ax1.set_title('📈 频谱通量趋势变化图', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-1, n_files + 5)

    # ---------- 子图2: 柱状图 ----------
    ax2 = fig.add_subplot(2, 1, 2)

    # 根据阈值确定颜色
    def get_bar_color(val):
        if np.isnan(val):
            return '#AAAAAA'
        elif val < 0.01:
            return '#90EE90'  # 浅绿（过于平稳）
        elif val < 0.1:
            return '#228B22'  # 深绿（理想范围）
        elif val < 0.5:
            return '#FFA500'  # 橙色（有节奏）
        else:
            return '#FF4500'  # 红橙（跳变）

    colors = [get_bar_color(f) for f in mean_fluxes]

    # 绘制柱状图
    bar_width = max(0.4, min(0.85, 40 / n_files))
    bars = ax2.bar(x, mean_fluxes, width=bar_width, color=colors,
                   alpha=0.88, edgecolor='#333333', linewidth=0.4)

    # 智能X轴标签
    if n_files <= 20:
        ax2.set_xticks(x)
        ax2.set_xticklabels(filenames, rotation=50, ha='right', fontsize=8)
    elif n_files <= 50:
        step = 2
        tick_idx = list(range(0, n_files, step))
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([filenames[i][:22] for i in tick_idx],
                            rotation=50, ha='right', fontsize=7)
    else:
        step = max(2, n_files // 25)
        tick_idx = list(range(0, n_files, step))
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([filenames[i][:18] for i in tick_idx],
                            rotation=50, ha='right', fontsize=6)

    # 柱状图的Y轴范围动态适配
    if len(valid_data) > 0:
        bar_data_min = np.nanmin(mean_fluxes)
        bar_data_max = np.nanmax(mean_fluxes)
        bar_data_range = bar_data_max - bar_data_min

        # 如果数据范围太小，扩展范围以显示差异
        if bar_data_range < bar_data_max * 0.05:
            bar_data_min = bar_data_min - bar_data_max * 0.1
            bar_data_max = bar_data_max + bar_data_max * 0.1

        bar_y_padding = bar_data_range * 0.1
        bar_y_bottom = max(0, bar_data_min - bar_y_padding)
        bar_y_top = bar_data_max + bar_y_padding

        ax2.set_ylim(bar_y_bottom, bar_y_top)

    ax2.set_xlabel('文件名', fontsize=11, fontweight='bold')
    ax2.set_ylabel('平均频谱通量', fontsize=11, fontweight='bold')
    ax2.set_title('📊 各文件频谱通量柱状对比图', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax2.set_xlim(-0.8, n_files - 0.2)

    # ==================== 说明文字（透明背景）====================
    description_text = (
        "【频谱通量 Spectral Flux】\n"
        "\n"
        "含义：相邻帧频谱变化幅度\n"
        "用途：检测节奏一致性或AI跳变\n"
        "公式：Flux = √[Σ|A_t(f) - A_{t-1}(f)|²]\n"
        "\n"
        "━━━━━━ 阈值参考 ━━━━━━\n"
        "● <0.01 浅绿：过于平稳/缺乏表现力\n"
        "● 0.01~0.1 深绿：平稳过渡 ✓ 理想\n"
        "● 0.1~0.5 橙色：节奏明显/有变化\n"
        "● >0.5 红色：频繁跳变 ⚠ 警惕"
    )

    fig.text(0.01, 0.01, description_text,
             fontsize=9,
             verticalalignment='bottom',
             horizontalalignment='left',
             linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.8',
                       facecolor='none',  # 透明背景
                       edgecolor='#666666',
                       linewidth=1.2))

    # ==================== 统计摘要 ====================
    if len(valid_data) > 0:
        # 找最优（最接近0.05，理想中点）和最差
        ideal_val = 0.05
        valid_indices = np.where(valid_mask)[0]
        best_local_idx = np.argmin(np.abs(valid_data - ideal_val))
        best_idx = valid_indices[best_local_idx]
        worst_idx = valid_indices[np.argmax(valid_data)]

        stats_text = (
            f"【统计摘要】\n"
            f"\n"
            f"文件总数：{n_files}\n"
            f"有效分析：{np.sum(valid_mask)}\n"
            f"\n"
            f"平均值：{np.nanmean(mean_fluxes):.4f}\n"
            f"标准差：{np.nanstd(mean_fluxes):.4f}\n"
            f"最小值：{np.nanmin(mean_fluxes):.4f}\n"
            f"最大值：{np.nanmax(mean_fluxes):.4f}\n"
            f"\n"
            f"━━━━ 推荐 ━━━━\n"
            f"最优：#{best_idx + 1}\n"
            f"  {filenames[best_idx][:18]}\n"
            f"  值={mean_fluxes[best_idx]:.4f}"
        )

        fig.text(0.99, 0.01, stats_text,
                 fontsize=9,
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 linespacing=1.4,
                 bbox=dict(boxstyle='round,pad=0.8',
                           facecolor='none',  # 透明背景
                           edgecolor='#666666',
                           linewidth=1.2))

    # 布局调整
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, hspace=0.38, left=0.07, right=0.93, top=0.95)

    print("📊 图表生成完成，弹出窗口展示中...")
    plt.show()

    # 返回分析结果
    return {
        'filenames': filenames,
        'mean_flux': mean_fluxes.tolist(),
        'p5_flux': p5_fluxes.tolist(),
        'p95_flux': p95_fluxes.tolist(),
        'valid': valid_mask.tolist(),
        'best_index': int(best_idx) if len(valid_data) > 0 else None,
        'stats': {
            'mean': float(np.nanmean(mean_fluxes)),
            'std': float(np.nanstd(mean_fluxes)),
            'min': float(np.nanmin(mean_fluxes)),
            'max': float(np.nanmax(mean_fluxes)),
        }
    }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 方式1：命令行调用
    import sys

    if len(sys.argv) > 1:
        wav_files = sys.argv[1:]
        result = analyze_spectral_flux(wav_files)
    else:
        # 方式2：直接在代码中指定路径
        print("=" * 50)
        print("使用方法:")
        print("  python spectral_flux.py file1.wav file2.wav ...")
        print("")
        print("或在Python中调用:")
        print("  from spectral_flux import analyze_spectral_flux")
        print("  result = analyze_spectral_flux(['a.wav', 'b.wav', ...])")
        print("=" * 50)

        # 示例（请替换为实际路径）
        # wav_files = [
        #     r"D:\models\epoch_100.wav",
        #     r"D:\models\epoch_200.wav",
        #     r"D:\models\epoch_300.wav",
        # ]
        # result = analyze_spectral_flux(wav_files)