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

+ 谐噪比（HNR）
    + 含义：谐波/噪声能量比，公式：HNR = 10 × log₁₀ (P_harmonic / P_noise)；检测气声、底噪或AI合成噪声。
    + 大于 20 dB：⭐ 干净自然
    + 10–20 dB：可接受
    + 小于 10 dB：噪声明显
"""

# pip install numpy matplotlib praat-parselmouth

"""
AI翻唱音频质量评估 - 谐噪比(HNR)分析

依赖安装:
    pip install numpy matplotlib praat-parselmouth

使用方法:
    from hnr_analyzer import analyze_hnr_quality

    wav_files = ["path/to/file1.wav", "path/to/file2.wav", ...]
    results = analyze_hnr_quality(wav_files)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple


def analyze_hnr_quality(wav_paths: List[str]) -> Dict[str, Optional[float]]:
    """
    分析多个WAV文件的谐噪比(HNR)并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组，已按模型轮数排序

    返回:
        Dict[str, Optional[float]]: 文件名到HNR值的映射
    """

    # 延迟导入，确保依赖检查在函数内部
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        raise ImportError("请先安装 praat-parselmouth: pip install praat-parselmouth")

    def calculate_hnr_single(wav_path: str) -> Tuple[str, Optional[float], Optional[str]]:
        """计算单个文件的HNR"""
        try:
            if not os.path.exists(wav_path):
                return (wav_path, None, "文件不存在")

            sound = parselmouth.Sound(wav_path)
            # 使用Praat的自相关方法计算HNR
            # 参数: time_step=0.01, minimum_pitch=75Hz, silence_threshold=0.1, periods_per_window=1.0
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_mean = call(harmonicity, "Get mean", 0, 0)

            # 检查是否为有效数值
            if hnr_mean is None or np.isnan(hnr_mean) or np.isinf(hnr_mean):
                return (wav_path, None, "无法计算HNR（可能是静音或纯噪声）")

            return (wav_path, float(hnr_mean), None)
        except Exception as e:
            return (wav_path, None, str(e))

    if not wav_paths:
        print("错误: 未提供任何文件路径")
        return {}

    # 并发处理所有文件
    results: Dict[str, Optional[float]] = {}
    num_workers = min(os.cpu_count() or 4, len(wav_paths), 8)

    print(f"{'=' * 60}")
    print(f"谐噪比(HNR)分析")
    print(f"{'=' * 60}")
    print(f"待分析文件: {len(wav_paths)} 个")
    print(f"并发线程数: {num_workers}")
    print(f"{'=' * 60}\n")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(calculate_hnr_single, path): path
                          for path in wav_paths}
        completed = 0
        for future in as_completed(future_to_path):
            path, hnr, error = future.result()
            completed += 1
            filename = os.path.basename(path)
            if error:
                print(f"  [{completed:3d}/{len(wav_paths)}] ⚠ {filename}: {error}")
            else:
                status = "🟢" if hnr >= 20 else ("🟡" if hnr >= 10 else "🔴")
                print(f"  [{completed:3d}/{len(wav_paths)}] {status} {filename}: {hnr:.2f} dB")
            results[path] = hnr

    # 按原始顺序整理结果
    hnr_values = [results.get(path) for path in wav_paths]
    file_names = [os.path.basename(path) for path in wav_paths]

    # 过滤有效数据
    valid_data = [(i, name, hnr) for i, (name, hnr) in enumerate(zip(file_names, hnr_values))
                  if hnr is not None]

    if not valid_data:
        print("\n错误: 没有有效的HNR数据，无法生成图表")
        return dict(zip(file_names, hnr_values))

    indices, names, values = zip(*valid_data)
    indices = list(indices)
    names = list(names)
    values = np.array(values)

    print(f"\n{'=' * 60}")
    print(f"分析完成! 有效文件: {len(valid_data)}/{len(wav_paths)}")
    print(f"{'=' * 60}\n")

    # ==================== 可视化 ====================

    # 设置中文字体（使用常规sans-serif字体）
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei',  # Windows
        'SimHei',  # Windows
        'PingFang SC',  # macOS
        'Hiragino Sans GB',  # macOS
        'WenQuanYi Micro Hei',  # Linux
        'Noto Sans CJK SC',  # Linux
        'DejaVu Sans',  # 通用后备
        'Arial'  # 最后后备
    ]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # 计算合适的图表尺寸
    num_files = len(wav_paths)
    fig_width = max(16, min(num_files * 0.25, 50))
    fig_height = 11

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')

    # 绘制折线图（显示趋势）
    ax.plot(indices, values, color='#3498db', linewidth=2, alpha=0.8, zorder=3,
            marker='', label='HNR变化趋势')

    # 根据阈值着色散点
    colors = []
    for v in values:
        if v >= 20:
            colors.append('#27ae60')  # 绿色 - 优秀
        elif v >= 10:
            colors.append('#f39c12')  # 橙色 - 可接受
        else:
            colors.append('#e74c3c')  # 红色 - 噪声明显

    # 绘制散点
    scatter = ax.scatter(indices, values, c=colors, s=80, zorder=5,
                         edgecolors='white', linewidth=1.2)

    # 动态计算Y轴范围，确保差异可见
    value_range = values.max() - values.min()
    if value_range < 3:  # 如果差异很小，放大显示
        center = (values.max() + values.min()) / 2
        y_min = center - 4
        y_max = center + 4
    else:
        margin = value_range * 0.2
        y_min = values.min() - margin
        y_max = values.max() + margin

    # 确保阈值线和关键区域可见
    y_min = min(y_min, 7)
    y_max = max(y_max, 23)
    ax.set_ylim(y_min, y_max)

    # 添加阈值线
    ax.axhline(y=20, color='#27ae60', linestyle='--', linewidth=2.5, alpha=0.9)
    ax.axhline(y=10, color='#f39c12', linestyle='--', linewidth=2.5, alpha=0.9)

    # 在阈值线旁边添加标签
    ax.text(len(wav_paths) - 0.5, 20.3, '⭐ 干净自然 (20 dB)', fontsize=9,
            color='#27ae60', ha='right', va='bottom', fontweight='bold')
    ax.text(len(wav_paths) - 0.5, 10.3, '可接受 (10 dB)', fontsize=9,
            color='#f39c12', ha='right', va='bottom', fontweight='bold')

    # 填充背景区域（质量分区）
    ax.axhspan(20, y_max, alpha=0.08, color='#27ae60', zorder=0)
    ax.axhspan(10, 20, alpha=0.08, color='#f39c12', zorder=0)
    ax.axhspan(y_min, 10, alpha=0.08, color='#e74c3c', zorder=0)

    # 设置X轴
    ax.set_xlim(-0.5, len(wav_paths) - 0.5)
    ax.set_xticks(list(range(len(wav_paths))))

    # 根据文件数量调整标签显示策略
    if num_files > 60:
        step = max(1, num_files // 25)
        visible_set = set(range(0, num_files, step))
        visible_set.add(0)
        visible_set.add(num_files - 1)
        labels = [file_names[i] if i in visible_set else '' for i in range(len(file_names))]
        rotation = 90
        fontsize = 7
    elif num_files > 30:
        step = max(1, num_files // 15)
        visible_set = set(range(0, num_files, step))
        visible_set.add(0)
        visible_set.add(num_files - 1)
        labels = [file_names[i] if i in visible_set else '' for i in range(len(file_names))]
        rotation = 70
        fontsize = 8
    else:
        labels = file_names
        rotation = 45
        fontsize = 9

    ax.set_xticklabels(labels, rotation=rotation, ha='right', fontsize=fontsize)

    # 设置标题和轴标签
    ax.set_xlabel('文件 (按模型训练轮数排序 →)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('谐噪比 HNR (dB)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('AI翻唱音频质量评估 — 谐噪比(HNR)分析',
                 fontsize=18, fontweight='bold', pad=20, color='#2c3e50')

    # 添加指标说明文字框（透明背景）
    info_text = (
        "谐噪比 (Harmonics-to-Noise Ratio)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "公式:\n"
        "  HNR = 10 × log₁₀(P_harmonic / P_noise)\n\n"
        "用途:\n"
        "  检测气声、底噪或AI合成噪声\n\n"
        "质量标准:\n"
        "  🟢 ≥ 20 dB → 干净自然\n"
        "  🟡 10~20 dB → 可接受\n"
        "  🔴 < 10 dB → 噪声明显"
    )

    ax.text(
        0.015, 0.97, info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='left',
        linespacing=1.5,
        bbox=dict(
            boxstyle='round,pad=0.7',
            facecolor='none',  # 透明背景
            edgecolor='#95a5a6',
            linewidth=1.5
        )
    )

    # 添加统计信息
    excellent_count = sum(1 for v in values if v >= 20)
    acceptable_count = sum(1 for v in values if 10 <= v < 20)
    poor_count = sum(1 for v in values if v < 10)
    total_valid = len(values)

    # 找出最佳和最差的文件
    best_idx = np.argmax(values)
    worst_idx = np.argmin(values)
    best_name = names[best_idx] if len(names[best_idx]) <= 25 else names[best_idx][:22] + "..."
    worst_name = names[worst_idx] if len(names[worst_idx]) <= 25 else names[worst_idx][:22] + "..."

    stats_text = (
        f"统计信息\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"有效文件: {len(valid_data)} / {len(wav_paths)}\n\n"
        f"数值统计:\n"
        f"  平均值: {values.mean():.2f} dB\n"
        f"  最大值: {values.max():.2f} dB\n"
        f"  最小值: {values.min():.2f} dB\n"
        f"  标准差: {values.std():.2f} dB\n\n"
        f"质量分布:\n"
        f"  🟢 优秀: {excellent_count} ({excellent_count / total_valid * 100:.1f}%)\n"
        f"  🟡 可接受: {acceptable_count} ({acceptable_count / total_valid * 100:.1f}%)\n"
        f"  🔴 较差: {poor_count} ({poor_count / total_valid * 100:.1f}%)\n\n"
        f"最佳: {best_name}\n"
        f"最差: {worst_name}"
    )

    ax.text(
        0.985, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        linespacing=1.5,
        bbox=dict(
            boxstyle='round,pad=0.7',
            facecolor='none',  # 透明背景
            edgecolor='#95a5a6',
            linewidth=1.5
        )
    )

    # 标记最佳和最差点
    ax.annotate(f'最佳\n{values[best_idx]:.1f}dB',
                xy=(indices[best_idx], values[best_idx]),
                xytext=(indices[best_idx], values[best_idx] + (y_max - y_min) * 0.08),
                fontsize=9, ha='center', color='#27ae60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

    ax.annotate(f'最差\n{values[worst_idx]:.1f}dB',
                xy=(indices[worst_idx], values[worst_idx]),
                xytext=(indices[worst_idx], values[worst_idx] - (y_max - y_min) * 0.08),
                fontsize=9, ha='center', color='#e74c3c', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    # 网格线
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#bdc3c7')
    ax.set_axisbelow(True)

    # 添加趋势说明
    if len(values) > 5:
        # 计算简单线性趋势
        z = np.polyfit(range(len(values)), values, 1)
        trend = "上升 📈" if z[0] > 0.01 else ("下降 📉" if z[0] < -0.01 else "平稳 ➡️")
        trend_text = f"整体趋势: {trend} (斜率: {z[0]:.3f} dB/轮)"
        ax.text(0.5, 0.02, trend_text, transform=ax.transAxes, fontsize=11,
                ha='center', va='bottom', color='#34495e', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#95a5a6', alpha=0.9))

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # 弹出窗口显示图表
    plt.show()

    # 返回结果
    return dict(zip(file_names, hnr_values))


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import sys

    # 示例：如果从命令行传入目录路径
    if len(sys.argv) > 1:
        import glob

        directory = sys.argv[1]
        wav_files = sorted(glob.glob(os.path.join(directory, "*.wav")))
        if wav_files:
            results = analyze_hnr_quality(wav_files)
        else:
            print(f"在 {directory} 中未找到WAV文件")
    else:
        print("=" * 60)
        print("AI翻唱音频质量评估 - 谐噪比(HNR)分析")
        print("=" * 60)
        print("\n使用方法:")
        print("  from hnr_analyzer import analyze_hnr_quality")
        print("  ")
        print("  wav_files = [")
        print('      "path/to/model_epoch100.wav",')
        print('      "path/to/model_epoch200.wav",')
        print('      "path/to/model_epoch300.wav",')
        print("      ...")
        print("  ]")
        print("  results = analyze_hnr_quality(wav_files)")
        print("\n或从命令行运行:")
        print("  python hnr_analyzer.py /path/to/wav/directory")