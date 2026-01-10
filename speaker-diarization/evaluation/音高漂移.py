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

+ Pitch Drift（音高漂移）
    + 含义：长音中F0缓慢偏移，AI翻唱常见问题，导致听感“音慢慢跑掉”。
    + DDSP / Diffusion 推理步数不足
    + 最小偏移：优秀
    + 明显偏移：差
    + 长音保持稳定性，斜率符合自然演唱曲线。
"""

# pip install numpy matplotlib scipy praat-parselmouth

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')


def analyze_pitch_drift(wav_paths: List[str]) -> List[Dict]:
    """
    分析AI翻唱WAV文件的Pitch Drift（音高漂移）指标并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数排序）

    返回:
        包含每个文件分析结果的字典列表
    """

    # ==================== 依赖导入 ====================
    try:
        import parselmouth
        from parselmouth.praat import call
    except ImportError:
        raise ImportError("请安装 praat-parselmouth: pip install praat-parselmouth")

    try:
        from scipy import stats
    except ImportError:
        raise ImportError("请安装 scipy: pip install scipy")

    # ==================== 单文件处理函数 ====================
    def process_single_file(args: tuple) -> Dict:
        """处理单个WAV文件，计算Pitch Drift指标"""
        idx, wav_path = args

        try:
            # 使用Praat提取F0（更准确）
            snd = parselmouth.Sound(wav_path)
            pitch = call(snd, "To Pitch", 0.0, 75, 600)

            # 获取F0序列
            f0_values = []
            times = []
            num_frames = call(pitch, "Get number of frames")

            for i in range(1, num_frames + 1):
                t = call(pitch, "Get time from frame number", i)
                f0 = call(pitch, "Get value in frame", i, "Hertz")
                if not np.isnan(f0) and f0 > 0:
                    f0_values.append(f0)
                    times.append(t)

            f0_values = np.array(f0_values)
            times = np.array(times)

            if len(f0_values) < 10:
                return {
                    'index': idx,
                    'path': wav_path,
                    'name': Path(wav_path).stem,
                    'drift_score': np.nan,
                    'avg_drift': np.nan,
                    'max_drift': np.nan,
                    'num_segments': 0,
                    'error': '有效音高帧数不足'
                }

            # 识别长音片段（连续有声区域）
            min_duration = 0.15  # 最小长音持续时间（秒）
            max_gap = 0.03  # 最大允许间隔（秒）

            segments = []
            seg_start_idx = 0

            for i in range(1, len(times)):
                gap = times[i] - times[i - 1]
                if gap > max_gap:
                    if times[i - 1] - times[seg_start_idx] >= min_duration:
                        segments.append((seg_start_idx, i))
                    seg_start_idx = i

            # 处理最后一个片段
            if len(times) > 0 and times[-1] - times[seg_start_idx] >= min_duration:
                segments.append((seg_start_idx, len(times)))

            # 计算每个长音片段的漂移
            drift_values = []

            for start_idx, end_idx in segments:
                seg_f0 = f0_values[start_idx:end_idx]
                seg_times = times[start_idx:end_idx]

                if len(seg_f0) < 3:
                    continue

                # 转换为音分（cents），相对于片段平均值
                mean_f0 = np.mean(seg_f0)
                seg_f0_cents = 1200 * np.log2(seg_f0 / mean_f0)

                # 线性回归计算斜率（cents/秒）
                rel_times = seg_times - seg_times[0]
                slope, _, _, _, _ = stats.linregress(rel_times, seg_f0_cents)

                drift_values.append(abs(slope))

            if not drift_values:
                return {
                    'index': idx,
                    'path': wav_path,
                    'name': Path(wav_path).stem,
                    'drift_score': np.nan,
                    'avg_drift': np.nan,
                    'max_drift': np.nan,
                    'num_segments': 0,
                    'error': '未找到有效长音片段'
                }

            # 计算综合漂移分数
            avg_drift = float(np.mean(drift_values))
            max_drift = float(np.max(drift_values))
            drift_score = 0.6 * avg_drift + 0.4 * max_drift

            return {
                'index': idx,
                'path': wav_path,
                'name': Path(wav_path).stem,
                'drift_score': drift_score,
                'avg_drift': avg_drift,
                'max_drift': max_drift,
                'num_segments': len(drift_values),
                'error': None
            }

        except Exception as e:
            return {
                'index': idx,
                'path': wav_path,
                'name': Path(wav_path).stem,
                'drift_score': np.nan,
                'avg_drift': np.nan,
                'max_drift': np.nan,
                'num_segments': 0,
                'error': str(e)
            }

    # ==================== 并发处理所有文件 ====================
    print(f"开始分析 {len(wav_paths)} 个文件...")
    results = []

    with ThreadPoolExecutor(max_workers=min(8, len(wav_paths))) as executor:
        futures = list(executor.map(process_single_file, enumerate(wav_paths)))
        results = sorted(futures, key=lambda x: x['index'])

    print("分析完成，正在生成图表...")

    # ==================== 可视化 ====================
    # 设置中文字体（常规字体，非等宽）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

    # 提取数据
    names = [r['name'] for r in results]
    drift_scores = np.array([r['drift_score'] for r in results], dtype=float)
    avg_drifts = np.array([r['avg_drift'] for r in results], dtype=float)
    max_drifts = np.array([r['max_drift'] for r in results], dtype=float)

    # 计算有效数据的范围
    valid_mask = ~np.isnan(drift_scores)
    valid_scores = drift_scores[valid_mask]

    if len(valid_scores) == 0:
        print("警告：没有有效的分析结果")
        return results

    score_min = np.min(valid_scores)
    score_max = np.max(valid_scores)
    score_range = score_max - score_min if score_max > score_min else 1

    # 动态调整Y轴范围以放大差异
    y_padding = score_range * 0.15
    y_min = max(0, score_min - y_padding)
    y_max = score_max + y_padding

    num_files = len(wav_paths)
    x = np.arange(num_files)

    # 颜色归一化
    norm_scores = np.zeros_like(drift_scores)
    norm_scores[valid_mask] = (drift_scores[valid_mask] - score_min) / score_range
    norm_scores[~valid_mask] = 0.5

    # ==================== 创建图表 ====================
    fig = plt.figure(figsize=(20, 14))

    # 根据文件数量调整布局
    if num_files <= 40:
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[3, 1],
                              hspace=0.25, wspace=0.15)
        ax_main = fig.add_subplot(gs[0, :])
        ax_bar = fig.add_subplot(gs[1, 0])
        ax_dist = fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.8], width_ratios=[3, 1],
                              hspace=0.25, wspace=0.15)
        ax_main = fig.add_subplot(gs[0, :])
        ax_bar = fig.add_subplot(gs[1, 0])
        ax_dist = fig.add_subplot(gs[1, 1])
        ax_heatmap = fig.add_subplot(gs[2, :])

    # ===== 主图：折线图显示趋势 =====
    ax_main.plot(x[valid_mask], drift_scores[valid_mask],
                 color='steelblue', linewidth=1.8, alpha=0.7, zorder=1, label='漂移趋势')

    scatter = ax_main.scatter(x[valid_mask], drift_scores[valid_mask],
                              c=drift_scores[valid_mask], cmap='RdYlGn_r',
                              s=100, edgecolors='white', linewidths=1.5, zorder=2,
                              vmin=score_min, vmax=score_max)

    # 标记无效点
    invalid_mask = ~valid_mask
    if np.any(invalid_mask):
        ax_main.scatter(x[invalid_mask], [y_min] * np.sum(invalid_mask),
                        color='gray', marker='x', s=60, label='分析失败', zorder=3)

    # 添加阈值参考线
    thresholds = [
        (10, '优秀 (≤10)', '#2ecc71', '-'),
        (25, '良好 (≤25)', '#9acd32', '--'),
        (40, '一般 (≤40)', '#f39c12', '--'),
        (60, '较差 (>40)', '#e74c3c', ':'),
    ]

    for thresh_val, thresh_name, thresh_color, thresh_style in thresholds:
        if thresh_val <= y_max * 1.5:
            ax_main.axhline(y=thresh_val, color=thresh_color, linestyle=thresh_style,
                            alpha=0.8, linewidth=2, label=thresh_name)

    # 填充优秀区域
    ax_main.axhspan(y_min, min(10, y_max), alpha=0.1, color='green')

    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlim(-0.5, num_files - 0.5)

    # X轴标签处理
    if num_files <= 30:
        ax_main.set_xticks(x)
        ax_main.set_xticklabels(names, rotation=55, ha='right', fontsize=8)
    elif num_files <= 60:
        step = 2
        ax_main.set_xticks(x[::step])
        ax_main.set_xticklabels([names[i] for i in range(0, num_files, step)],
                                rotation=55, ha='right', fontsize=7)
    else:
        step = max(1, num_files // 25)
        ax_main.set_xticks(x[::step])
        ax_main.set_xticklabels([names[i] for i in range(0, num_files, step)],
                                rotation=55, ha='right', fontsize=6)

    ax_main.set_xlabel('文件（按模型训练轮数递增排序 →）', fontsize=12)
    ax_main.set_ylabel('Pitch Drift 分数 (cents/秒)', fontsize=12)
    ax_main.set_title('AI翻唱音高漂移 (Pitch Drift) 质量分析', fontsize=16, fontweight='bold', pad=15)
    ax_main.legend(loc='upper right', fontsize=9, ncol=2)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax_main, shrink=0.8, pad=0.02)
    cbar.set_label('漂移程度 (cents/秒)\n← 优秀 | 较差 →', fontsize=10)

    # 添加说明文字（背景透明）
    desc_text = (
        "【Pitch Drift 音高漂移指标说明】\n\n"
        "● 含义：长音中F0（基频）缓慢偏移\n"
        "● 表现：听感上「音慢慢跑掉」\n"
        "● 原因：DDSP/Diffusion推理步数不足\n"
        "● 单位：cents/秒（音分每秒）\n\n"
        "● 评判标准（数值越小越好）：\n"
        "   ✓ 优秀：≤ 10 cents/s\n"
        "   ○ 良好：≤ 25 cents/s\n"
        "   △ 一般：≤ 40 cents/s\n"
        "   ✗ 较差：> 40 cents/s\n\n"
        "● 理想状态：长音保持稳定，\n"
        "   斜率符合自然演唱曲线"
    )

    text_box = ax_main.text(0.01, 0.99, desc_text, transform=ax_main.transAxes, fontsize=9,
                            verticalalignment='top', fontfamily='sans-serif',
                            linespacing=1.3,
                            bbox=dict(boxstyle='round,pad=0.6', facecolor='none',
                                      edgecolor='#cccccc', alpha=1.0, linewidth=1.5))

    # ===== 条形图对比 =====
    bar_colors = [plt.cm.RdYlGn_r(ns) if not np.isnan(drift_scores[i]) else '#cccccc'
                  for i, ns in enumerate(norm_scores)]
    bars = ax_bar.bar(x, np.nan_to_num(drift_scores, nan=0), color=bar_colors,
                      edgecolor='white', linewidth=0.3)

    ax_bar.set_ylim(y_min, y_max)
    ax_bar.set_xlim(-0.5, num_files - 0.5)

    # 标注最佳和最差
    if len(valid_scores) >= 2:
        best_idx = int(np.nanargmin(drift_scores))
        worst_idx = int(np.nanargmax(drift_scores))

        # 最佳标注
        ax_bar.annotate(f'最佳\n{names[best_idx][:15]}\n{drift_scores[best_idx]:.1f}',
                        xy=(best_idx, drift_scores[best_idx]),
                        xytext=(best_idx, max(drift_scores[best_idx] + score_range * 0.25, y_min + score_range * 0.3)),
                        ha='center', fontsize=8, color='#27ae60', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='#27ae60'))

        # 最差标注
        ax_bar.annotate(f'最差\n{names[worst_idx][:15]}\n{drift_scores[worst_idx]:.1f}',
                        xy=(worst_idx, drift_scores[worst_idx]),
                        xytext=(worst_idx, min(drift_scores[worst_idx] - score_range * 0.2, y_max - score_range * 0.1)),
                        ha='center', fontsize=8, color='#e74c3c', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='#e74c3c'))

    ax_bar.set_xlabel('文件索引', fontsize=11)
    ax_bar.set_ylabel('Pitch Drift (cents/秒)', fontsize=11)
    ax_bar.set_title('各文件漂移分数对比', fontsize=12, fontweight='bold')
    ax_bar.grid(True, alpha=0.3, axis='y')

    # ===== 分布直方图 =====
    ax_dist.hist(valid_scores, bins=min(20, len(valid_scores) // 2 + 1),
                 color='steelblue', edgecolor='white', alpha=0.7, orientation='horizontal')
    ax_dist.axhline(y=np.mean(valid_scores), color='red', linestyle='--',
                    linewidth=2, label=f'平均值: {np.mean(valid_scores):.1f}')
    ax_dist.axhline(y=np.median(valid_scores), color='orange', linestyle='-',
                    linewidth=2, label=f'中位数: {np.median(valid_scores):.1f}')

    ax_dist.set_ylim(y_min, y_max)
    ax_dist.set_xlabel('文件数量', fontsize=11)
    ax_dist.set_title('分数分布', fontsize=12, fontweight='bold')
    ax_dist.legend(loc='upper right', fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    # 统计信息
    stats_text = (
        f"【统计信息】\n"
        f"─────────────\n"
        f"文件总数: {num_files}\n"
        f"有效分析: {len(valid_scores)}\n"
        f"─────────────\n"
        f"平均值: {np.mean(valid_scores):.2f}\n"
        f"中位数: {np.median(valid_scores):.2f}\n"
        f"标准差: {np.std(valid_scores):.2f}\n"
        f"─────────────\n"
        f"最小值: {np.min(valid_scores):.2f}\n"
        f"最大值: {np.max(valid_scores):.2f}\n"
        f"─────────────\n"
        f"优秀(≤10): {np.sum(valid_scores <= 10)}\n"
        f"良好(≤25): {np.sum((valid_scores > 10) & (valid_scores <= 25))}\n"
        f"一般(≤40): {np.sum((valid_scores > 25) & (valid_scores <= 40))}\n"
        f"较差(>40): {np.sum(valid_scores > 40)}"
    )

    ax_dist.text(0.95, 0.02, stats_text, transform=ax_dist.transAxes, fontsize=9,
                 verticalalignment='bottom', horizontalalignment='right',
                 fontfamily='sans-serif', linespacing=1.2,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='none',
                           edgecolor='#cccccc', alpha=1.0))

    # ===== 热力图（文件数量多时显示）=====
    if num_files > 40:
        # 创建热力图数据
        heatmap_data = drift_scores.reshape(1, -1)
        im = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r',
                               vmin=score_min, vmax=score_max)

        ax_heatmap.set_yticks([])
        ax_heatmap.set_xlabel('文件索引（按模型轮数递增）', fontsize=11)
        ax_heatmap.set_title('漂移程度热力图（绿色=优秀，红色=较差）', fontsize=12, fontweight='bold')

        # 添加文件名标记
        step = max(1, num_files // 20)
        ax_heatmap.set_xticks(range(0, num_files, step))
        ax_heatmap.set_xticklabels([names[i][:12] for i in range(0, num_files, step)],
                                   rotation=45, ha='right', fontsize=7)

        plt.colorbar(im, ax=ax_heatmap, orientation='vertical', shrink=0.8, pad=0.02)

    plt.tight_layout()
    plt.show()

    # 打印分析结果摘要
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)
    print(f"文件总数: {num_files}")
    print(f"有效分析: {len(valid_scores)}")
    print(f"平均漂移: {np.mean(valid_scores):.2f} cents/秒")
    print(f"最佳文件: {names[int(np.nanargmin(drift_scores))]} ({np.min(valid_scores):.2f})")
    print(f"最差文件: {names[int(np.nanargmax(drift_scores))]} ({np.max(valid_scores):.2f})")
    print("=" * 60)

    return results


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例：传入WAV文件路径列表
    wav_files = [
        "path/to/model_epoch_100.wav",
        "path/to/model_epoch_200.wav",
        "path/to/model_epoch_300.wav",
        # ... 更多文件
    ]

    results = analyze_pitch_drift(wav_files)