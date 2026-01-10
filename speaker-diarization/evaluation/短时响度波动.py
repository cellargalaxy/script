"""
1. 我打算使用以下指标对ai翻唱的wav文件进行质量评价。
2. 我只有使用ai翻唱出来的多个wav文件，我能提供这些文件的路径。
3. 判断以下指标，只有wav文件路径，这些文件之间是否能对比出优劣，如果对比不出优劣就不需要再继续了
4. 如果能对比出优劣，写一个python函数，入参是wav文件路径的字符串数组
5. 该python函数实现以下指标的计算，并且将计算结果画为图表进行可视化对比
6. 图表的类型，需要根据指标的特殊进行选择，目的是能更加直观的看出各个wav文件的优劣
7. 图表的数轴标度，为了避免不同文件之间的指标差异过小，在图中看不出区别，需要更加明显的处理
8. 文件大约有几十个，需要合理排版，以能清晰看出每个文件的数据走向与图标
9. 并且文件路径数组已经排好序，模型的轮数是递增的。
10. 在图表中增加该指标的文字描述，阈值的辅助信息，图表使用常规字体而不是等宽字体
11. 尽量将代码都收敛到函数内部，方便调用
12. 最后提供一个完整可用的python函数，以及其需要安装的依赖

+ 短时响度波动（Short-term Loudness Variance）
    + 含义：短时间窗（如3秒）内响度变化程度，反映情感表达的动态性；用于判断是否“全程一个音量”（情感死板）或压缩过度。
    + 适中（方差适度）：自然起伏，富有情感
    + 波动太小：情感死板
    + 波动太大：不稳定，破音风险
"""

# pip install numpy librosa matplotlib scipy

"""
AI翻唱音频质量分析 - 短时响度波动 (Short-term Loudness Variance)
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os


def analyze_short_term_loudness_variance(
        wav_paths: List[str],
        window_sec: float = 3.0,
        save_path: Optional[str] = None,
        show_plot: bool = True
) -> Dict:
    """
    分析AI翻唱wav文件的短时响度波动（Short-term Loudness Variance）

    参数:
        wav_paths: wav文件路径列表
        window_sec: 短时窗口长度（秒），默认3秒
        save_path: 图表保存路径，默认None不保存
        show_plot: 是否显示图表，默认True

    返回:
        results: 包含各文件分析结果的字典
    """

    # ==================== 配置字体 ====================
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC',
                                   'Hiragino Sans GB', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # ==================== 数据处理 ====================
    results = {}

    for path in wav_paths:
        if not os.path.exists(path):
            print(f"⚠️ 警告: 文件不存在 - {path}")
            continue

        try:
            # 加载音频
            y, sr = librosa.load(path, sr=None, mono=True)

            # 计算帧级RMS（100ms帧，50ms跳跃）
            frame_length = int(0.1 * sr)
            hop_length = int(0.05 * sr)

            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = librosa.amplitude_to_db(rms + 1e-10, ref=np.max(rms) if np.max(rms) > 0 else 1.0)

            # 计算短时响度（窗口内平均）
            window_frames = max(1, int(window_sec / (hop_length / sr)))
            hop_frames = max(1, window_frames // 2)

            short_term_loudness = []
            time_points = []

            for i in range(0, len(rms_db) - window_frames + 1, hop_frames):
                window = rms_db[i:i + window_frames]
                short_term_loudness.append(np.mean(window))
                time_points.append((i + window_frames / 2) * hop_length / sr)

            if len(short_term_loudness) < 2:
                print(f"⚠️ 警告: 音频过短，跳过 - {path}")
                continue

            short_term_loudness = np.array(short_term_loudness)
            time_points = np.array(time_points)

            # 计算统计量
            variance = float(np.var(short_term_loudness))
            std = float(np.std(short_term_loudness))
            mean_loudness = float(np.mean(short_term_loudness))
            dynamic_range = float(np.ptp(short_term_loudness))

            filename = os.path.basename(path)
            results[filename] = {
                'path': path,
                'short_term_loudness': short_term_loudness,
                'time_points': time_points,
                'variance': variance,
                'std': std,
                'mean': mean_loudness,
                'dynamic_range': dynamic_range,
                'duration': len(y) / sr
            }

        except Exception as e:
            print(f"❌ 处理失败 {path}: {e}")
            continue

    if not results:
        print("❌ 没有成功处理任何文件")
        return {}

    # ==================== 可视化 ====================
    _create_loudness_charts(results, window_sec, save_path, show_plot)

    return results


def _get_rating(variance: float) -> tuple:
    """根据方差值返回评级和颜色"""
    if variance < 5:
        return "情感死板", "#3498db", "波动过小"
    elif variance <= 25:
        return "自然起伏 ✓", "#27ae60", "良好"
    elif variance <= 50:
        return "波动较大", "#f39c12", "需注意"
    else:
        return "不稳定", "#e74c3c", "风险高"


def _create_loudness_charts(results: Dict, window_sec: float,
                            save_path: Optional[str], show_plot: bool):
    """创建可视化图表"""

    filenames = list(results.keys())
    n_files = len(filenames)

    # 提取数据
    variances = [results[f]['variance'] for f in filenames]
    stds = [results[f]['std'] for f in filenames]

    # 颜色方案
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_files, 8)))[:n_files]

    # 创建图表布局
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.32, wspace=0.22)

    # 简化文件名显示
    def short_name(name, max_len=16):
        return name[:max_len - 2] + '..' if len(name) > max_len else name

    short_names = [short_name(f) for f in filenames]

    # ============ 图1: 方差对比（核心指标）============
    ax1 = fig.add_subplot(gs[0, 0])

    x_pos = np.arange(n_files)
    bar_colors = [_get_rating(v)[1] for v in variances]
    bars1 = ax1.bar(x_pos, variances, color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    # 阈值区域（背景色块）
    y_max = max(max(variances) * 1.35, 55)
    ax1.axhspan(0, 5, alpha=0.12, color='#3498db')
    ax1.axhspan(5, 25, alpha=0.12, color='#27ae60')
    ax1.axhspan(25, 50, alpha=0.12, color='#f39c12')
    ax1.axhspan(50, y_max, alpha=0.12, color='#e74c3c')

    # 阈值线
    ax1.axhline(y=5, color='#3498db', linestyle='--', linewidth=2, label='下限 (5 dB²)')
    ax1.axhline(y=25, color='#27ae60', linestyle='--', linewidth=2, label='良好上限 (25 dB²)')
    ax1.axhline(y=50, color='#e74c3c', linestyle='--', linewidth=2, label='风险线 (50 dB²)')

    # 动态调整Y轴范围（放大差异）
    if len(set(variances)) > 1:
        var_range = max(variances) - min(variances)
        y_min = max(0, min(variances) - var_range * 0.2)
        y_max = max(variances) + var_range * 0.3
        # 确保阈值线可见
        y_max = max(y_max, 30)
    else:
        y_min, y_max = 0, max(variances) * 1.5
    ax1.set_ylim(y_min, y_max)

    # 数值标签
    for bar, var in zip(bars1, variances):
        ax1.annotate(f'{var:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha='center',
                     fontsize=11, fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names, rotation=40, ha='right', fontsize=9)
    ax1.set_ylabel('方差 (dB²)', fontsize=11, fontweight='bold')
    ax1.set_title('📊 短时响度方差对比（核心指标）', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # ============ 图2: 动态范围对比 ============
    ax2 = fig.add_subplot(gs[0, 1])

    dynamic_ranges = [results[f]['dynamic_range'] for f in filenames]
    bars2 = ax2.bar(x_pos, dynamic_ranges, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

    # 动态调整Y轴
    if len(set(dynamic_ranges)) > 1:
        dr_range = max(dynamic_ranges) - min(dynamic_ranges)
        y_min_dr = max(0, min(dynamic_ranges) - dr_range * 0.15)
        y_max_dr = max(dynamic_ranges) + dr_range * 0.25
    else:
        y_min_dr, y_max_dr = 0, max(dynamic_ranges) * 1.3
    ax2.set_ylim(y_min_dr, y_max_dr)

    for bar, dr in zip(bars2, dynamic_ranges):
        ax2.annotate(f'{dr:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha='center',
                     fontsize=11, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, rotation=40, ha='right', fontsize=9)
    ax2.set_ylabel('动态范围 (dB)', fontsize=11, fontweight='bold')
    ax2.set_title('📈 响度动态范围对比', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # ============ 图3: 箱线图分布 ============
    ax3 = fig.add_subplot(gs[1, 0])

    box_data = [results[f]['short_term_loudness'] for f in filenames]
    bp = ax3.boxplot(box_data, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.2)
    for median in bp['medians']:
        median.set(color='darkred', linewidth=2)

    ax3.set_xticklabels(short_names, rotation=40, ha='right', fontsize=9)
    ax3.set_ylabel('响度 (dB)', fontsize=11, fontweight='bold')
    ax3.set_title('📦 短时响度分布（箱线图）', fontsize=13, fontweight='bold', pad=10)
    ax3.grid(axis='y', alpha=0.3, linestyle=':')

    # ============ 图4: 时间序列曲线 ============
    ax4 = fig.add_subplot(gs[1, 1])

    for idx, filename in enumerate(filenames):
        data = results[filename]
        label = short_name(filename, 18)
        ax4.plot(data['time_points'], data['short_term_loudness'],
                 color=colors[idx], linewidth=1.8, alpha=0.85, label=label)

    ax4.set_xlabel('时间 (秒)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('响度 (dB)', fontsize=11, fontweight='bold')
    ax4.set_title(f'📉 短时响度时间曲线 (窗口={window_sec}s)', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax4.grid(True, alpha=0.3, linestyle=':')

    # ============ 图5: 说明与结果面板 ============
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    # 指标说明区域
    desc_text = """【指标说明】短时响度波动 (Short-term Loudness Variance)

定义：在短时间窗（{}秒）内，响度变化的程度，反映情感表达的动态性。
用途：判断翻唱是否「全程一个音量」（情感死板）或动态失控（破音风险）。

评判标准：
  • 方差 < 5 dB²     → 波动过小，情感死板，缺乏表现力
  • 方差 5~25 dB²   → 适中良好，自然起伏，富有情感 ✓
  • 方差 25~50 dB²  → 波动较大，情感夸张或录音问题
  • 方差 > 50 dB²    → 波动过大，不稳定，存在破音风险""".format(window_sec)

    ax5.text(0.02, 0.98, desc_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='sans-serif',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f0f8ff',
                       edgecolor='#4a90d9', alpha=0.95, linewidth=1.5))

    # 结果汇总
    result_lines = ["【分析结果汇总】\n"]
    for filename in filenames:
        data = results[filename]
        rating, color, level = _get_rating(data['variance'])
        result_lines.append(
            f"  {filename[:28]:28s}  │  方差: {data['variance']:6.2f} dB²  │  "
            f"标准差: {data['std']:5.2f} dB  │  动态范围: {data['dynamic_range']:5.1f} dB  │  "
            f"评级: {rating}"
        )

    result_text = '\n'.join(result_lines)
    ax5.text(0.52, 0.98, result_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='sans-serif',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffef0',
                       edgecolor='#d4a017', alpha=0.95, linewidth=1.5))

    # 总标题
    fig.suptitle('🎵 AI翻唱音频质量分析 — 短时响度波动',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ 图表已保存: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 替换为你的wav文件路径
    wav_files = [
        r"path/to/song1.wav",
        r"path/to/song2.wav",
        r"path/to/song3.wav",
    ]

    results = analyze_short_term_loudness_variance(
        wav_paths=wav_files,
        window_sec=3.0,
        save_path="loudness_variance_analysis.png",
        show_plot=True
    )

    # 打印数值结果
    print("\n" + "=" * 60)
    print("数值结果:")
    print("=" * 60)
    for filename, data in results.items():
        rating, _, _ = _get_rating(data['variance'])
        print(f"\n📁 {filename}")
        print(f"   方差: {data['variance']:.2f} dB²")
        print(f"   标准差: {data['std']:.2f} dB")
        print(f"   动态范围: {data['dynamic_range']:.2f} dB")
        print(f"   评级: {rating}")