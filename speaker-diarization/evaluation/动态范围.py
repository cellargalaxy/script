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

+ RMS Dynamic Range（RMS 动态范围）
    + 含义：RMS最大与最小值差异，公式：DR = 20 × log₁₀(最大振幅 / 最小可听振幅)；判断表达起伏。
    + 大于60 dB：丰富动态
    + 40–60 dB：正常
    + 20–40 dB：可能压缩
    + 小于20 dB：扁平，<10 dB过度压缩
    + 流行唱法：12-18 dB；艺术歌曲：>20 dB。
"""

# pip install numpy librosa matplotlib

"""
AI翻唱WAV文件质量评价 - RMS动态范围分析

依赖安装:
    pip install numpy librosa matplotlib

使用示例:
    from rms_analyzer import analyze_rms_dynamic_range
    wav_files = ["model_100.wav", "model_200.wav", ...]
    analyze_rms_dynamic_range(wav_files)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


def analyze_rms_dynamic_range(wav_paths: List[str]) -> None:
    """
    分析多个WAV文件的RMS动态范围并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数递增排序）
    """

    # ==================== 内部导入依赖 ====================
    try:
        import librosa
    except ImportError:
        raise ImportError("请先安装librosa: pip install librosa")

    # ==================== 单文件处理函数 ====================
    def calculate_single_file(wav_path: str) -> Tuple[str, float, dict]:
        """计算单个文件的RMS动态范围"""
        filename = os.path.basename(wav_path)
        try:
            # 加载音频文件
            y, sr = librosa.load(wav_path, sr=None)

            # 计算RMS能量（分帧）
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

            # 过滤静音部分（保留超过最大值1%的帧，避免静音干扰）
            threshold = np.max(rms) * 0.01
            rms_filtered = rms[rms > threshold]

            if len(rms_filtered) < 2:
                rms_filtered = rms[rms > 1e-10]

            if len(rms_filtered) == 0:
                return filename, 0.0, {'valid': False}

            rms_max = np.max(rms_filtered)
            rms_min = np.min(rms_filtered)

            # 计算动态范围: DR = 20 × log₁₀(RMS_max / RMS_min)
            dr = 20 * np.log10(rms_max / rms_min) if rms_min > 0 else 0.0

            return filename, dr, {
                'valid': True,
                'rms_max': rms_max,
                'rms_min': rms_min,
                'duration': len(y) / sr
            }

        except Exception as e:
            print(f"⚠ 处理文件 {filename} 时出错: {e}")
            return filename, float('nan'), {'valid': False, 'error': str(e)}

    # ==================== 并发处理所有文件 ====================
    print(f"📂 开始处理 {len(wav_paths)} 个WAV文件...")

    max_workers = min(8, os.cpu_count() or 4, len(wav_paths))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(calculate_single_file, wav_paths))

    print("✅ 音频分析完成，正在生成图表...")

    # 提取结果
    filenames = [r[0] for r in results]
    dr_values = [r[1] for r in results]

    # ==================== 图表配置 ====================
    # 设置中文字体（优先使用非等宽字体）
    font_candidates = [
        'Microsoft YaHei',  # Windows
        'PingFang SC',  # macOS
        'Noto Sans CJK SC',  # Linux
        'SimHei',  # 备选
        'Arial Unicode MS',  # 跨平台
        'DejaVu Sans'  # 最后备选
    ]
    plt.rcParams['font.sans-serif'] = font_candidates
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # 根据文件数量动态调整图表尺寸
    n_files = len(filenames)
    fig_width = max(14, min(50, n_files * 0.35))
    fig_height = 10

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    fig.patch.set_facecolor('white')

    x = np.arange(n_files)

    # ==================== 根据动态范围值着色 ====================
    def get_color(v):
        """根据动态范围值返回对应颜色"""
        if np.isnan(v) or v < 10:
            return '#E74C3C'  # 红色 - 过度压缩
        elif v < 20:
            return '#E67E22'  # 橙色 - 扁平
        elif v < 40:
            return '#F39C12'  # 金色 - 可能压缩
        elif v < 60:
            return '#3498DB'  # 蓝色 - 正常
        else:
            return '#27AE60'  # 绿色 - 丰富动态

    colors = [get_color(v) for v in dr_values]

    # ==================== 绘制主图表 ====================
    # 折线（显示趋势）
    valid_mask = [not np.isnan(v) for v in dr_values]
    valid_x = [i for i, m in enumerate(valid_mask) if m]
    valid_y = [dr_values[i] for i in valid_x]

    if valid_y:
        ax.plot(valid_x, valid_y, linewidth=1.2, color='#7F8C8D',
                alpha=0.6, zorder=1, linestyle='-')

    # 散点（颜色区分质量等级）
    ax.scatter(x, dr_values, c=colors, s=60, zorder=3,
               edgecolors='white', linewidths=0.8)

    # ==================== 添加阈值参考线 ====================
    threshold_config = [
        (60, '#27AE60', '丰富动态 (>60 dB)'),
        (40, '#3498DB', '正常范围 (40-60 dB)'),
        (20, '#E67E22', '可能压缩 (20-40 dB)'),
        (10, '#E74C3C', '过度压缩 (<10 dB)'),
    ]

    for val, color, label in threshold_config:
        ax.axhline(y=val, color=color, linestyle='--',
                   alpha=0.7, linewidth=1.5, label=label)

    # ==================== 动态调整Y轴范围（放大差异）====================
    valid_values = [v for v in dr_values if not np.isnan(v)]
    if valid_values:
        data_min = min(valid_values)
        data_max = max(valid_values)
        data_range = data_max - data_min

        # 留出适当边距，同时确保能看到关键阈值线
        padding = max(data_range * 0.15, 3)
        y_min = max(0, data_min - padding)
        y_max = max(data_max + padding, 65)  # 确保能看到60dB阈值

        ax.set_ylim(y_min, y_max)

    # ==================== X轴标签处理 ====================
    ax.set_xticks(x)

    # 智能显示标签：文件多时抽样显示
    if n_files > 60:
        step = max(1, n_files // 25)
        visible_indices = list(range(0, n_files, step))
        if (n_files - 1) not in visible_indices:
            visible_indices.append(n_files - 1)

        labels = [filenames[i] if i in visible_indices else '' for i in range(n_files)]
        ax.set_xticklabels(labels, rotation=90, fontsize=7, ha='center')
    elif n_files > 30:
        ax.set_xticklabels(filenames, rotation=90, fontsize=7, ha='center')
    else:
        ax.set_xticklabels(filenames, rotation=45, fontsize=8, ha='right')

    # ==================== 轴标签和标题 ====================
    ax.set_xlabel('文件名（按模型轮数递增 →）', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMS 动态范围 (dB)', fontsize=12, fontweight='bold')
    ax.set_title('AI翻唱WAV文件 · RMS动态范围对比分析',
                 fontsize=15, fontweight='bold', pad=15)

    # ==================== 指标说明文字（透明背景）====================
    desc_text = """【RMS动态范围】

公式: DR = 20 × log₁₀(RMS_max / RMS_min)

判断标准:
  ● 大于 60 dB → 丰富动态 (优秀)
  ● 40 ~ 60 dB → 正常范围 (良好)
  ● 20 ~ 40 dB → 可能压缩 (一般)
  ● 10 ~ 20 dB → 较扁平   (较差)
  ● 小于 10 dB → 过度压缩 (差)

唱法参考:
  ● 流行唱法: 12~18 dB
  ● 艺术歌曲: >20 dB"""

    ax.text(
        0.02, 0.97, desc_text,
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=9,
        family='sans-serif',
        linespacing=1.3,
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='none',  # 透明背景
            edgecolor='#BDC3C7',
            linewidth=1
        )
    )

    # ==================== 图例 ====================
    legend = ax.legend(
        loc='upper right',
        fontsize=9,
        framealpha=0.95,
        edgecolor='#BDC3C7',
        title='阈值参考线',
        title_fontsize=10
    )

    # ==================== 网格 ====================
    ax.grid(True, alpha=0.3, linestyle='-', which='major', axis='y')
    ax.grid(True, alpha=0.15, linestyle=':', which='major', axis='x')

    # ==================== 底部统计信息 ====================
    if valid_values:
        stats_text = (
            f"📊 统计信息:  "
            f"最小值 = {min(valid_values):.2f} dB  |  "
            f"最大值 = {max(valid_values):.2f} dB  |  "
            f"均值 = {np.mean(valid_values):.2f} dB  |  "
            f"标准差 = {np.std(valid_values):.2f} dB  |  "
            f"有效文件数 = {len(valid_values)}/{n_files}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                 style='italic', color='#2C3E50')

    # ==================== 最终布局调整 ====================
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.93)

    print("📈 图表生成完成，正在显示...")
    plt.show()


# ==================== 测试/示例代码 ====================
if __name__ == '__main__':
    import sys

    # 示例：从命令行参数读取文件路径
    if len(sys.argv) > 1:
        wav_files = sys.argv[1:]
        analyze_rms_dynamic_range(wav_files)
    else:
        print("使用方法:")
        print("  python rms_analyzer.py file1.wav file2.wav ...")
        print("")
        print("或在Python中导入使用:")
        print("  from rms_analyzer import analyze_rms_dynamic_range")
        print("  analyze_rms_dynamic_range(['file1.wav', 'file2.wav', ...])")