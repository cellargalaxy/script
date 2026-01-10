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

+ MFCC距离分析
    + 含义：相邻帧MFCC余弦距离；检测音色突变。
    + 小于0.3（连续3帧）：连贯
    + 大于0.3：不连贯
"""

# pip install numpy librosa matplotlib scipy

"""
MFCC距离分析 - AI翻唱质量评估工具

依赖安装：
pip install numpy librosa matplotlib scipy

或创建 requirements.txt:
numpy>=1.20.0
librosa>=0.9.0
matplotlib>=3.5.0
scipy>=1.7.0
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import List, Dict, Tuple
import warnings


def _compute_single_file_mfcc(wav_path: str) -> Dict:
    """
    计算单个文件的MFCC距离指标（内部函数，用于并发处理）
    """
    try:
        warnings.filterwarnings('ignore')

        # 加载音频
        y, sr = librosa.load(wav_path, sr=22050)

        # 提取MFCC特征 (13维)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # 计算相邻帧的余弦距离
        distances = []
        for i in range(mfcc.shape[1] - 1):
            frame1 = mfcc[:, i]
            frame2 = mfcc[:, i + 1]

            norm1 = np.linalg.norm(frame1)
            norm2 = np.linalg.norm(frame2)

            if norm1 > 1e-10 and norm2 > 1e-10:
                # 余弦距离 = 1 - 余弦相似度
                cos_sim = np.dot(frame1, frame2) / (norm1 * norm2)
                dist = 1 - cos_sim
                distances.append(max(0, min(2, dist)))  # 限制在合理范围

        distances = np.array(distances)
        threshold = 0.3

        if len(distances) == 0:
            return {
                'path': wav_path,
                'filename': os.path.basename(wav_path),
                'success': False,
                'error': '无法提取有效的MFCC特征'
            }

        # 计算超过阈值的比例
        exceed_ratio = np.sum(distances > threshold) / len(distances)

        # 计算连续超过阈值的段落数（连续3帧及以上）
        consecutive_exceed = 0
        current_streak = 0
        for d in distances:
            if d > threshold:
                current_streak += 1
            else:
                if current_streak >= 3:
                    consecutive_exceed += 1
                current_streak = 0
        if current_streak >= 3:
            consecutive_exceed += 1

        # 计算音频时长
        duration = len(y) / sr

        return {
            'path': wav_path,
            'filename': os.path.basename(wav_path),
            'mean_distance': float(np.mean(distances)),
            'max_distance': float(np.max(distances)),
            'std_distance': float(np.std(distances)),
            'median_distance': float(np.median(distances)),
            'exceed_ratio': float(exceed_ratio),
            'consecutive_exceed': int(consecutive_exceed),
            'total_frames': len(distances),
            'duration': float(duration),
            'success': True
        }

    except Exception as e:
        return {
            'path': wav_path,
            'filename': os.path.basename(wav_path),
            'success': False,
            'error': str(e)
        }


def _calculate_dynamic_y_limits(data: np.ndarray, margin_factor: float = 0.15) -> Tuple[float, float]:
    """
    计算动态Y轴范围，确保数据差异明显可见

    参数:
        data: 数据数组
        margin_factor: 边距因子（数据范围的百分比）

    返回:
        (y_min, y_max): 合适的Y轴范围
    """
    if len(data) == 0:
        return 0.0, 1.0

    y_min = np.min(data)
    y_max = np.max(data)
    y_range = y_max - y_min

    # 如果数据范围太小，扩大显示范围以突出差异
    if y_range < 0.01:  # 差异很小的情况
        y_center = (y_min + y_max) / 2
        y_min = y_center - 0.02
        y_max = y_center + 0.02
    else:
        # 添加合适的边距
        y_margin = max(y_range * margin_factor, 0.01)  # 至少0.01的边距
        y_min = max(0, y_min - y_margin)
        y_max = y_max + y_margin

    # 确保最小值不会小于0
    y_min = max(0, y_min)

    return float(y_min), float(y_max)


def analyze_mfcc_distance(wav_paths: List[str]) -> None:
    """
    分析多个WAV文件的MFCC距离，评估AI翻唱质量并可视化对比

    MFCC距离分析说明：
    - 含义：相邻帧MFCC余弦距离，用于检测音色突变
    - 阈值：小于0.3（连续3帧）表示连贯，大于0.3表示不连贯
    - 所有指标越低越好

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数递增排序）
    """

    if not wav_paths:
        print("错误：文件路径列表为空")
        return

    # ==================== 设置字体 ====================
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'PingFang SC',
        'Hiragino Sans GB', 'WenQuanYi Micro Hei', 'Arial Unicode MS',
        'Noto Sans CJK SC', 'Source Han Sans CN', 'DejaVu Sans'
    ]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # ==================== 并发处理文件 ====================
    print(f"开始分析 {len(wav_paths)} 个WAV文件...")
    print("=" * 60)

    results = [None] * len(wav_paths)
    max_workers = min(os.cpu_count() or 4, 8, len(wav_paths))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_compute_single_file_mfcc, path): idx
            for idx, path in enumerate(wav_paths)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result
            completed += 1

            status = "✓" if result['success'] else "✗"
            print(f"[{completed:3d}/{len(wav_paths)}] {status} {result['filename']}")

    # ==================== 过滤有效结果 ====================
    valid_results = [r for r in results if r and r['success']]
    failed_results = [r for r in results if r and not r['success']]

    print("=" * 60)
    print(f"处理完成: 成功 {len(valid_results)}, 失败 {len(failed_results)}")

    if failed_results:
        print("\n失败文件:")
        for r in failed_results:
            print(f"  - {r['filename']}: {r.get('error', '未知错误')}")

    if not valid_results:
        print("\n错误：没有成功处理的文件")
        return

    # ==================== 提取数据 ====================
    filenames = [r['filename'] for r in valid_results]
    mean_distances = np.array([r['mean_distance'] for r in valid_results])
    max_distances = np.array([r['max_distance'] for r in valid_results])
    std_distances = np.array([r['std_distance'] for r in valid_results])
    exceed_ratios = np.array([r['exceed_ratio'] for r in valid_results])
    consecutive_exceeds = np.array([r['consecutive_exceed'] for r in valid_results])

    n_files = len(valid_results)
    indices = np.arange(n_files)

    # ==================== 简化文件名 ====================
    def shorten_name(name: str, max_len: int = 18) -> str:
        name = os.path.splitext(name)[0]
        if len(name) > max_len:
            return '...' + name[-(max_len - 3):]
        return name

    short_names = [shorten_name(f) for f in filenames]

    # ==================== 计算统计信息 ====================
    best_idx = int(np.argmin(mean_distances))
    worst_idx = int(np.argmax(mean_distances))

    # ==================== 创建图表 ====================
    # 动态调整图表大小
    fig_width = max(16, min(28, n_files * 0.25))
    fig_height = 16

    fig = plt.figure(figsize=(fig_width, fig_height))

    # 总标题
    fig.suptitle('MFCC距离分析 - AI翻唱质量评估',
                 fontsize=18, fontweight='bold', y=0.98)

    # 指标说明文本（透明背景）
    desc_text = (
        '【指标说明】 MFCC距离：相邻帧梅尔频率倒谱系数的余弦距离，用于检测音色突变\n'
        '【阈值标准】 距离 < 0.3（连续3帧以上）= 音色连贯  |  距离 > 0.3 = 存在音色突变\n'
        '【评判原则】 所有指标越低越好，表示音色过渡越平滑自然'
    )
    fig.text(0.5, 0.935, desc_text, ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='none',
                       edgecolor='#3498db', linewidth=1.5, alpha=0.8),
             linespacing=1.5)

    # 创建子图布局
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3],
                          hspace=0.38, wspace=0.20,
                          left=0.06, right=0.96, top=0.87, bottom=0.10)

    # 设置x轴刻度的显示策略
    if n_files <= 25:
        x_step = 1
        rotation = 45
        fontsize = 8
    elif n_files <= 50:
        x_step = 2
        rotation = 45
        fontsize = 7
    else:
        x_step = max(1, n_files // 25)
        rotation = 60
        fontsize = 6

    # ==================== 图1: 平均MFCC距离 ====================
    ax1 = fig.add_subplot(gs[0, 0])
    colors1 = ['#27ae60' if d < 0.25 else '#f39c12' if d < 0.35 else '#e74c3c'
               for d in mean_distances]
    bars1 = ax1.bar(indices, mean_distances, color=colors1, alpha=0.85, width=0.8,
                    edgecolor='white', linewidth=0.5)

    # 动态计算Y轴范围（不显示阈值线）
    y_min1, y_max1 = _calculate_dynamic_y_limits(mean_distances, margin_factor=0.2)
    ax1.set_ylim(y_min1, y_max1)

    ax1.set_ylabel('距离值', fontsize=11)
    ax1.set_title('① 平均MFCC距离（越低越好）', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xticks(indices[::x_step])
    ax1.set_xticklabels([short_names[i] for i in range(0, n_files, x_step)],
                        rotation=rotation, ha='right', fontsize=fontsize)

    # ==================== 图2: 最大MFCC距离 ====================
    ax2 = fig.add_subplot(gs[0, 1])
    colors2 = ['#27ae60' if d < 0.5 else '#f39c12' if d < 0.8 else '#e74c3c'
               for d in max_distances]
    ax2.bar(indices, max_distances, color=colors2, alpha=0.85, width=0.8,
            edgecolor='white', linewidth=0.5)

    # 动态计算Y轴范围（不显示阈值线）
    y_min2, y_max2 = _calculate_dynamic_y_limits(max_distances, margin_factor=0.15)
    ax2.set_ylim(y_min2, y_max2)

    ax2.set_ylabel('距离值', fontsize=11)
    ax2.set_title('② 最大MFCC距离（越低越好）', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_xticks(indices[::x_step])
    ax2.set_xticklabels([short_names[i] for i in range(0, n_files, x_step)],
                        rotation=rotation, ha='right', fontsize=fontsize)

    # ==================== 图3: 超过阈值的比例 ====================
    ax3 = fig.add_subplot(gs[1, 0])
    exceed_pct = exceed_ratios * 100
    colors3 = ['#27ae60' if p < 10 else '#f39c12' if p < 25 else '#e74c3c'
               for p in exceed_pct]
    ax3.bar(indices, exceed_pct, color=colors3, alpha=0.85, width=0.8,
            edgecolor='white', linewidth=0.5)

    # 动态计算Y轴范围（不显示阈值线）
    y_min3, y_max3 = _calculate_dynamic_y_limits(exceed_pct, margin_factor=0.15)
    ax3.set_ylim(y_min3, y_max3)

    ax3.set_ylabel('比例 (%)', fontsize=11)
    ax3.set_title('③ 超过阈值(0.3)的帧比例（越低越好）', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.set_xticks(indices[::x_step])
    ax3.set_xticklabels([short_names[i] for i in range(0, n_files, x_step)],
                        rotation=rotation, ha='right', fontsize=fontsize)

    # ==================== 图4: 连续音色突变段落数 ====================
    ax4 = fig.add_subplot(gs[1, 1])
    colors4 = ['#27ae60' if c == 0 else '#f39c12' if c <= 2 else '#e74c3c'
               for c in consecutive_exceeds]
    ax4.bar(indices, consecutive_exceeds, color=colors4, alpha=0.85, width=0.8,
            edgecolor='white', linewidth=0.5)

    # 动态计算Y轴范围
    if len(consecutive_exceeds) > 0:
        y_min4 = 0
        y_max4 = np.max(consecutive_exceeds)
        y_range4 = y_max4 - y_min4

        # 如果所有值都是0，设置合适的范围显示
        if y_range4 == 0:
            y_max4 = 2  # 显示0-2的范围以便观察
        else:
            # 添加边距，确保有足够的显示空间
            y_max4 = y_max4 + max(1, y_range4 * 0.2)

        ax4.set_ylim(y_min4, y_max4)

        # 设置整数刻度
        if y_max4 <= 10:
            ax4.set_yticks(np.arange(0, y_max4 + 1, 1))
        elif y_max4 <= 20:
            ax4.set_yticks(np.arange(0, y_max4 + 1, 2))
        else:
            step = max(1, int(y_max4 / 10))
            ax4.set_yticks(np.arange(0, y_max4 + 1, step))

    ax4.set_ylabel('段落数', fontsize=11)
    ax4.set_title('④ 连续音色突变段落数（≥3帧为一段，越少越好）',
                  fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.set_xticks(indices[::x_step])
    ax4.set_xticklabels([short_names[i] for i in range(0, n_files, x_step)],
                        rotation=rotation, ha='right', fontsize=fontsize)

    # ==================== 图5: 综合趋势图 ====================
    ax5 = fig.add_subplot(gs[2, :])

    # 绘制折线
    line1, = ax5.plot(indices, mean_distances, 'o-', color='#3498db',
                      label='平均距离', markersize=5, linewidth=1.8, alpha=0.9)

    # 填充区域表示质量等级（透明度降低，避免遮挡数据）
    ax5.fill_between(indices, 0, 0.25, alpha=0.05, color='#27ae60', label='优秀区域 (<0.25)')
    ax5.fill_between(indices, 0.25, 0.35, alpha=0.05, color='#f39c12', label='一般区域 (0.25-0.35)')

    # 添加趋势线
    if n_files >= 3:
        z = np.polyfit(indices, mean_distances, 1)
        p = np.poly1d(z)
        trend_line = p(indices)
        trend_direction = "↓ 改善" if z[0] < 0 else "↑ 退化" if z[0] > 0 else "→ 稳定"
        ax5.plot(indices, trend_line, '--', color='#9b59b6', linewidth=2.5, alpha=0.8,
                 label=f'趋势线 (斜率: {z[0]:.5f}) {trend_direction}')

    ax5.set_xlabel('文件序号（按模型训练轮数递增 →）', fontsize=12)
    ax5.set_ylabel('平均MFCC距离', fontsize=12)
    ax5.set_title('⑤ 训练进度趋势图（观察模型质量随训练轮数的变化）',
                  fontsize=13, fontweight='bold', pad=12)
    ax5.legend(loc='upper right', fontsize=9, ncol=2)
    ax5.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 动态Y轴范围
    y_min5, y_max5 = _calculate_dynamic_y_limits(mean_distances, margin_factor=0.2)
    ax5.set_ylim(y_min5, y_max5)

    # X轴刻度
    if n_files <= 30:
        ax5.set_xticks(indices)
        ax5.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    else:
        step5 = max(1, n_files // 30)
        ax5.set_xticks(indices[::step5])
        ax5.set_xticklabels([short_names[i] for i in range(0, n_files, step5)],
                            rotation=45, ha='right', fontsize=7)

    # ==================== 统计摘要 ====================
    summary_text = (
        f'【统计摘要】\n'
        f'文件总数: {n_files}  |  '
        f'平均距离均值: {np.mean(mean_distances):.4f}  |  '
        f'标准差: {np.std(mean_distances):.4f}\n'
        f'最佳: {filenames[best_idx]} (均值: {mean_distances[best_idx]:.4f})  |  '
        f'最差: {filenames[worst_idx]} (均值: {mean_distances[worst_idx]:.4f})'
    )

    fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='none',
                       edgecolor='#2c3e50', linewidth=1.5, alpha=0.8),
             linespacing=1.4)

    # ==================== 显示图表 ====================
    plt.show()

    # ==================== 打印详细结果 ====================
    print("\n" + "=" * 80)
    print("详细分析结果（按原始顺序）")
    print("=" * 80)
    print(f"{'序号':<5} {'文件名':<35} {'平均距离':<12} {'最大距离':<12} {'超阈值比例':<12} {'突变段落':<10}")
    print("-" * 80)

    for i, r in enumerate(valid_results):
        exceed_pct_val = r['exceed_ratio'] * 100
        quality = "优秀" if r['mean_distance'] < 0.25 else "良好" if r['mean_distance'] < 0.35 else "较差"
        print(f"{i + 1:<5} {r['filename']:<35} {r['mean_distance']:<12.4f} "
              f"{r['max_distance']:<12.4f} {exceed_pct_val:<11.2f}% {r['consecutive_exceed']:<10} [{quality}]")

    print("=" * 80)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例用法
    wav_files = [
        "path/to/model_epoch_100.wav",
        "path/to/model_epoch_200.wav",
        "path/to/model_epoch_300.wav",
        # ... 更多文件
    ]

    analyze_mfcc_distance(wav_files)