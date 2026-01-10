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

+ 削波检测（Clipping Detection）
    + 含义：检测样本值接近最大（≥0.99×最大值）的比例，公式：削波率 = (削波采样点数 / 总采样点数) × 100%；反映失真。
    + 0%：无削波
    + 小于0.1%：可接受
    + 0.1–1%：有问题
    + 大于1%：严重失真
"""

# pip install numpy scipy matplotlib

"""
AI翻唱WAV文件质量评估 - 削波检测（Clipping Detection）

依赖安装:
pip install numpy scipy matplotlib

使用示例:
    wav_files = ["path/to/file1.wav", "path/to/file2.wav", ...]
    analyze_clipping_detection(wav_files)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple


def analyze_clipping_detection(wav_paths: List[str]) -> Dict[str, Any]:
    """
    分析多个WAV文件的削波检测指标并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（按模型轮数递增排序）

    返回:
        包含分析结果的字典
    """

    # ================== 内部函数定义 ==================

    def calculate_clipping_rate(args: Tuple[int, str]) -> Dict[str, Any]:
        """计算单个文件的削波率"""
        index, wav_path = args
        try:
            sample_rate, data = wavfile.read(wav_path)

            # 立体声转单声道
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # 根据数据类型确定理论最大值
            dtype = data.dtype
            if dtype == np.int16:
                max_val = 32767.0
            elif dtype == np.int32:
                max_val = 2147483647.0
            elif dtype in [np.float32, np.float64]:
                max_val = 1.0
            elif np.issubdtype(dtype, np.integer):
                max_val = float(np.iinfo(dtype).max)
            else:
                max_val = float(np.max(np.abs(data))) if np.max(np.abs(data)) > 0 else 1.0

            # 归一化到 [-1, 1]
            data_normalized = data.astype(np.float64) / max_val

            # 计算削波（|样本值| >= 0.99 视为削波）
            clipping_threshold = 0.99
            clipping_samples = int(np.sum(np.abs(data_normalized) >= clipping_threshold))
            total_samples = len(data_normalized)
            clipping_rate = (clipping_samples / total_samples) * 100.0

            return {
                'index': index,
                'path': wav_path,
                'name': os.path.basename(wav_path),
                'clipping_rate': clipping_rate,
                'clipping_samples': clipping_samples,
                'total_samples': total_samples,
                'sample_rate': sample_rate,
                'duration': total_samples / sample_rate,
                'error': None
            }
        except Exception as e:
            return {
                'index': index,
                'path': wav_path,
                'name': os.path.basename(wav_path),
                'clipping_rate': None,
                'error': str(e)
            }

    def get_quality_color(rate: float) -> str:
        """根据削波率返回质量等级颜色"""
        if rate == 0:
            return '#27AE60'  # 绿色 - 无削波
        elif rate < 0.1:
            return '#3498DB'  # 蓝色 - 可接受
        elif rate < 1.0:
            return '#F39C12'  # 橙色 - 有问题
        else:
            return '#E74C3C'  # 红色 - 严重失真

    def get_quality_level(rate: float) -> str:
        """根据削波率返回质量等级文字"""
        if rate == 0:
            return '无削波'
        elif rate < 0.1:
            return '可接受'
        elif rate < 1.0:
            return '有问题'
        else:
            return '严重失真'

    # ================== 主处理逻辑 ==================

    if not wav_paths:
        print("错误: 文件路径列表为空")
        return {}

    n_total = len(wav_paths)
    print(f"正在并发分析 {n_total} 个WAV文件...")

    # 并发处理
    max_workers = min(n_total, (os.cpu_count() or 4) * 2, 16)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(calculate_clipping_rate, (i, path)): i
            for i, path in enumerate(wav_paths)
        }

        completed = 0
        for future in as_completed(future_to_args):
            results.append(future.result())
            completed += 1
            if completed % 10 == 0 or completed == n_total:
                print(f"  进度: {completed}/{n_total}")

    # 按原始顺序排序
    results.sort(key=lambda x: x['index'])

    # 分离有效和错误结果
    valid_results = [r for r in results if r['error'] is None]
    error_results = [r for r in results if r['error'] is not None]

    if error_results:
        print(f"\n⚠ 警告: {len(error_results)} 个文件处理失败:")
        for r in error_results[:5]:
            print(f"   [{r['index']}] {r['name']}: {r['error']}")
        if len(error_results) > 5:
            print(f"   ... 还有 {len(error_results) - 5} 个错误")

    if not valid_results:
        print("错误: 没有可分析的有效文件")
        return {}

    # 提取数据用于绑图
    indices = [r['index'] for r in valid_results]
    names = [r['name'] for r in valid_results]
    rates = [r['clipping_rate'] for r in valid_results]
    colors = [get_quality_color(r) for r in rates]

    n_valid = len(valid_results)

    # ================== 可视化 ==================

    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC',
                                       'Hiragino Sans GB', 'DejaVu Sans', 'Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # 动态计算图表尺寸
    fig_width = max(16, min(32, n_valid * 0.25 + 6))
    fig_height = 14

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    fig.suptitle('削波检测 (Clipping Detection) - AI翻唱WAV文件质量分析',
                 fontsize=16, fontweight='bold', y=0.98)

    # 创建子图布局
    gs = fig.add_gridspec(4, 1, height_ratios=[3.5, 2, 1.2, 0.8],
                          hspace=0.35, top=0.93, bottom=0.05)

    ax_line = fig.add_subplot(gs[0])  # 折线趋势图
    ax_bar = fig.add_subplot(gs[1])  # 条形分布图
    ax_stats = fig.add_subplot(gs[2])  # 统计信息
    ax_legend = fig.add_subplot(gs[3])  # 图例说明

    # ---------- 1. 折线趋势图 ----------

    # 绘制趋势线
    ax_line.plot(indices, rates, '-', linewidth=2, color='#2C3E50',
                 alpha=0.7, zorder=2, label='削波率趋势')

    # 散点（颜色表示质量）
    for i, (idx, rate, color) in enumerate(zip(indices, rates, colors)):
        ax_line.scatter([idx], [rate], c=[color], s=80, zorder=4,
                        edgecolors='white', linewidth=1.5)

    # 阈值线
    ax_line.axhline(y=0.1, color='#F39C12', linestyle='--', linewidth=2,
                    alpha=0.9, label='可接受阈值 (0.1%)')
    ax_line.axhline(y=1.0, color='#E74C3C', linestyle='--', linewidth=2,
                    alpha=0.9, label='严重失真阈值 (1.0%)')

    # 填充质量区域
    max_rate = max(rates)
    y_upper = max(max_rate * 1.4, 1.5, 0.2)

    ax_line.fill_between([min(indices) - 0.5, max(indices) + 0.5], 0, 0.1,
                         color='#3498DB', alpha=0.08, label='_可接受区')
    ax_line.fill_between([min(indices) - 0.5, max(indices) + 0.5], 0.1, 1.0,
                         color='#F39C12', alpha=0.08, label='_问题区')
    ax_line.fill_between([min(indices) - 0.5, max(indices) + 0.5], 1.0, y_upper,
                         color='#E74C3C', alpha=0.08, label='_严重区')

    # 动态调整Y轴以突出差异
    if max_rate == 0:
        y_max = 0.15
    elif max_rate < 0.05:
        y_max = max(0.12, max_rate * 2.5)
    elif max_rate < 0.2:
        y_max = max(0.3, max_rate * 1.8)
    elif max_rate < 1.0:
        y_max = max(1.2, max_rate * 1.4)
    else:
        y_max = max_rate * 1.3

    ax_line.set_ylim(0, y_max)
    ax_line.set_xlim(min(indices) - 0.5, max(indices) + 0.5)

    # X轴刻度优化
    if n_valid > 40:
        step = max(1, n_valid // 25)
        ticks = list(range(min(indices), max(indices) + 1, step))
        if max(indices) not in ticks:
            ticks.append(max(indices))
        ax_line.set_xticks(ticks)

    ax_line.set_xlabel('文件索引（按模型训练轮数递增）', fontsize=11)
    ax_line.set_ylabel('削波率 (%)', fontsize=11)
    ax_line.set_title('削波率变化趋势', fontsize=12, pad=10)
    ax_line.legend(loc='upper right', fontsize=9)
    ax_line.grid(True, alpha=0.3, linestyle='-')

    # 添加指标说明（透明背景）
    desc_box = (
        "【削波检测说明】\n"
        "定义：检测样本值接近最大值（≥0.99×最大值）的比例\n"
        "公式：削波率 = (削波采样点数 ÷ 总采样点数) × 100%\n"
        "意义：反映音频失真程度，削波率越低越好"
    )
    ax_line.text(0.02, 0.97, desc_box, transform=ax_line.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='none',
                           edgecolor='#BDC3C7', alpha=0.8, linewidth=1))

    # ---------- 2. 条形分布图 ----------

    bars = ax_bar.bar(indices, rates, color=colors, width=0.85,
                      edgecolor='white', linewidth=0.5)

    ax_bar.axhline(y=0.1, color='#F39C12', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_bar.axhline(y=1.0, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.8)

    ax_bar.set_xlim(min(indices) - 0.5, max(indices) + 0.5)
    ax_bar.set_ylim(0, y_max)

    if n_valid > 40:
        ax_bar.set_xticks(ticks)

    ax_bar.set_xlabel('文件索引', fontsize=10)
    ax_bar.set_ylabel('削波率 (%)', fontsize=10)
    ax_bar.set_title('各文件削波率对比（颜色=质量等级：绿色最佳→红色最差）', fontsize=11)
    ax_bar.grid(True, alpha=0.2, axis='y')

    # ---------- 3. 统计信息面板 ----------

    ax_stats.axis('off')

    # 计算统计量
    rates_arr = np.array(rates)
    no_clip = sum(1 for r in rates if r == 0)
    acceptable = sum(1 for r in rates if 0 < r < 0.1)
    problematic = sum(1 for r in rates if 0.1 <= r < 1.0)
    severe = sum(1 for r in rates if r >= 1.0)

    best_idx = int(np.argmin(rates_arr))
    worst_idx = int(np.argmax(rates_arr))

    # 构建统计文本
    stats_text = (
        f"╔══════════════════════════════════════════════════════════════════════════════════════╗\n"
        f"║  【基本统计】                                                                         ║\n"
        f"║    • 有效文件数: {n_valid}    • 削波率范围: {min(rates):.6f}% ~ {max(rates):.6f}%                  ║\n"
        f"║    • 平均削波率: {np.mean(rates):.6f}%    • 中位数: {np.median(rates):.6f}%    • 标准差: {np.std(rates):.6f}%     ║\n"
        f"╠══════════════════════════════════════════════════════════════════════════════════════╣\n"
        f"║  【质量分布】                                                                         ║\n"
        f"║    ● 无削波 (=0%): {no_clip:>3} 个 ({no_clip / n_valid * 100:.1f}%)        ● 可接受 (<0.1%): {acceptable:>3} 个 ({acceptable / n_valid * 100:.1f}%)       ║\n"
        f"║    ● 有问题 (0.1-1%): {problematic:>3} 个 ({problematic / n_valid * 100:.1f}%)      ● 严重失真 (>1%): {severe:>3} 个 ({severe / n_valid * 100:.1f}%)        ║\n"
        f"╠══════════════════════════════════════════════════════════════════════════════════════╣\n"
        f"║  【最佳文件】 索引[{indices[best_idx]}] {names[best_idx][:40]:<40} → {rates[best_idx]:.6f}%  ║\n"
        f"║  【最差文件】 索引[{indices[worst_idx]}] {names[worst_idx][:40]:<40} → {rates[worst_idx]:.6f}%  ║\n"
        f"╚══════════════════════════════════════════════════════════════════════════════════════╝"
    )

    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, verticalalignment='center', horizontalalignment='center',
                  family='monospace',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA',
                            edgecolor='#DEE2E6', alpha=0.95))

    # ---------- 4. 图例说明 ----------

    ax_legend.axis('off')

    # 阈值说明
    threshold_text = (
        "【评价阈值标准】   "
        "■ 0% = 无削波（最佳）   "
        "■ <0.1% = 可接受   "
        "■ 0.1%~1% = 有问题   "
        "■ >1% = 严重失真"
    )

    ax_legend.text(0.5, 0.6, threshold_text, transform=ax_legend.transAxes,
                   fontsize=10, ha='center', va='center')

    # 颜色图例
    legend_colors = [('#27AE60', '无削波'), ('#3498DB', '可接受'),
                     ('#F39C12', '有问题'), ('#E74C3C', '严重失真')]

    for i, (color, label) in enumerate(legend_colors):
        x_pos = 0.28 + i * 0.135
        ax_legend.add_patch(plt.Rectangle((x_pos, 0.45), 0.02, 0.25,
                                          transform=ax_legend.transAxes,
                                          facecolor=color, edgecolor='white',
                                          linewidth=1))

    plt.show()

    # ================== 控制台输出 ==================

    print("\n" + "=" * 70)
    print("分析完成！图表已弹出显示。")
    print("=" * 70)

    # 返回结果数据
    return {
        'valid_count': n_valid,
        'error_count': len(error_results),
        'results': valid_results,
        'statistics': {
            'min': float(np.min(rates)),
            'max': float(np.max(rates)),
            'mean': float(np.mean(rates)),
            'median': float(np.median(rates)),
            'std': float(np.std(rates)),
            'best_file': names[best_idx],
            'worst_file': names[worst_idx]
        },
        'quality_distribution': {
            'no_clipping': no_clip,
            'acceptable': acceptable,
            'problematic': problematic,
            'severe': severe
        }
    }


# ================== 使用示例 ==================

if __name__ == "__main__":
    import glob

    # 示例：获取文件夹下所有wav文件并排序
    # wav_files = sorted(glob.glob(r"C:\path\to\your\wav_files\*.wav"))

    # 或手动指定文件列表
    # wav_files = [
    #     r"C:\output\model_epoch_100.wav",
    #     r"C:\output\model_epoch_200.wav",
    #     r"C:\output\model_epoch_300.wav",
    #     # ... 更多文件
    # ]

    # 调用分析函数
    # result = analyze_clipping_detection(wav_files)

    print("请提供wav文件路径列表并调用 analyze_clipping_detection(wav_paths)")
