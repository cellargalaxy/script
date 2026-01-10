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

+ 频谱滚降点（Spectral Roll-off）
    + 含义：累积能量达85%的频率点；检测高频完整性，斜率突变表示人工痕迹。
    + 3000–8000 Hz：正常范围
    + 小于3000 Hz：高频损失
    + 大于10000 Hz：高频噪声
"""

# pip install numpy librosa matplotlib scipy

"""
频谱滚降点（Spectral Roll-off）分析工具
用于评估AI翻唱WAV文件的高频质量

依赖安装：
pip install numpy librosa matplotlib scipy
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def _compute_single_rolloff(args: Tuple[int, str, float]) -> Tuple[int, str, float, np.ndarray, float, float]:
    """
    计算单个文件的频谱滚降点（供并发调用）

    返回: (索引, 文件名, 平均滚降点, 滚降点序列, 标准差, 音频时长)
    """
    idx, wav_path, roll_percent = args
    try:
        y, sr = librosa.load(wav_path, sr=None)
        duration = len(y) / sr

        # 计算频谱滚降点
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, roll_percent=roll_percent
        ).flatten()

        mean_rolloff = float(np.mean(rolloff))
        std_rolloff = float(np.std(rolloff))

        return (idx, os.path.basename(wav_path), mean_rolloff, rolloff, std_rolloff, duration)

    except Exception as e:
        print(f"[错误] 处理文件失败: {wav_path}\n  原因: {e}")
        return (idx, os.path.basename(wav_path), np.nan, np.array([]), np.nan, 0.0)


def analyze_spectral_rolloff(
        wav_paths: List[str],
        roll_percent: float = 0.85,
        max_workers: Optional[int] = None
) -> dict:
    """
    分析多个WAV文件的频谱滚降点（Spectral Roll-off）并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数递增排序）
        roll_percent: 滚降点的能量百分比阈值，默认0.85（85%）
        max_workers: 最大并发数，默认为CPU核心数

    返回:
        包含分析结果的字典
    """

    if not wav_paths:
        raise ValueError("文件路径列表不能为空")

    # ==================== 并发计算 ====================
    print(f"开始分析 {len(wav_paths)} 个文件...")

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(wav_paths))

    # 准备参数
    task_args = [(i, path, roll_percent) for i, path in enumerate(wav_paths)]

    # 并发执行
    results = [None] * len(wav_paths)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_single_rolloff, args): args[0] for args in task_args}

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            idx = result[0]
            results[idx] = result
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"  进度: {completed}/{len(wav_paths)}")

    # 提取数据
    file_names = [r[1] for r in results]
    mean_rolloffs = np.array([r[2] for r in results])
    rolloff_series = [r[3] for r in results]
    std_rolloffs = np.array([r[4] for r in results])
    durations = np.array([r[5] for r in results])

    # ==================== 可视化配置 ====================
    # 设置中文字体
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # 定义阈值常量
    THRESHOLD_LOW = 3000  # 高频损失阈值
    THRESHOLD_NORMAL_HIGH = 8000  # 正常范围上限
    THRESHOLD_NOISE = 10000  # 高频噪声阈值

    # 创建颜色映射
    def get_quality_color(val):
        if np.isnan(val):
            return '#808080'  # 灰色 - 无效
        elif val < THRESHOLD_LOW:
            return '#E74C3C'  # 红色 - 高频损失
        elif val > THRESHOLD_NOISE:
            return '#E67E22'  # 橙色 - 高频噪声
        elif val <= THRESHOLD_NORMAL_HIGH:
            return '#27AE60'  # 绿色 - 正常范围
        else:
            return '#F39C12'  # 黄色 - 偏高但可接受

    colors = [get_quality_color(v) for v in mean_rolloffs]

    # ==================== 创建图表 ====================
    n_files = len(wav_paths)

    # 根据文件数量调整图表大小
    fig_width = max(16, min(24, n_files * 0.3))
    fig_height = 14

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle('频谱滚降点 (Spectral Roll-off) 质量分析报告',
                 fontsize=16, fontweight='bold', y=0.98)

    # 创建网格布局
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.8],
                          hspace=0.35, wspace=0.15,
                          left=0.06, right=0.94, top=0.92, bottom=0.08)

    # ==================== 图1: 主趋势图 ====================
    ax1 = fig.add_subplot(gs[0, :])

    x = np.arange(n_files)

    # 绘制柱状图
    bars = ax1.bar(x, mean_rolloffs, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=0.5, zorder=2)

    # 绘制趋势线
    valid_mask = ~np.isnan(mean_rolloffs)
    if np.sum(valid_mask) > 1:
        ax1.plot(x[valid_mask], mean_rolloffs[valid_mask],
                 'b-', linewidth=2, alpha=0.8, zorder=3, label='趋势线')
        ax1.scatter(x[valid_mask], mean_rolloffs[valid_mask],
                    c='blue', s=30, zorder=4, edgecolors='white', linewidths=0.5)

    # 绘制标准差范围（误差带）
    if np.any(valid_mask):
        ax1.fill_between(x,
                         mean_rolloffs - std_rolloffs,
                         mean_rolloffs + std_rolloffs,
                         alpha=0.2, color='blue', zorder=1, label='±1σ 范围')

    # 添加阈值参考线
    ax1.axhline(y=THRESHOLD_LOW, color='#E74C3C', linestyle='--',
                linewidth=2, alpha=0.8, zorder=5)
    ax1.axhline(y=THRESHOLD_NORMAL_HIGH, color='#27AE60', linestyle='--',
                linewidth=2, alpha=0.8, zorder=5)
    ax1.axhline(y=THRESHOLD_NOISE, color='#E67E22', linestyle='--',
                linewidth=2, alpha=0.8, zorder=5)

    # 绘制正常范围区域
    ax1.axhspan(THRESHOLD_LOW, THRESHOLD_NORMAL_HIGH,
                alpha=0.1, color='green', zorder=0)

    # 在右侧标注阈值
    ax1.text(n_files + 0.5, THRESHOLD_LOW, f'{THRESHOLD_LOW} Hz\n(高频损失线)',
             va='center', ha='left', fontsize=9, color='#E74C3C')
    ax1.text(n_files + 0.5, THRESHOLD_NORMAL_HIGH, f'{THRESHOLD_NORMAL_HIGH} Hz\n(正常上限)',
             va='center', ha='left', fontsize=9, color='#27AE60')
    ax1.text(n_files + 0.5, THRESHOLD_NOISE, f'{THRESHOLD_NOISE} Hz\n(噪声阈值)',
             va='center', ha='left', fontsize=9, color='#E67E22')

    # 设置坐标轴 - 修改X轴标签为文件名
    ax1.set_xlabel('文件名称', fontsize=11)
    ax1.set_ylabel('频谱滚降点 (Hz)', fontsize=11)
    ax1.set_title('平均频谱滚降点趋势对比', fontsize=13, fontweight='bold', pad=10)

    # 动态调整Y轴范围以突出差异
    valid_values = mean_rolloffs[valid_mask]
    if len(valid_values) > 0:
        data_min, data_max = np.min(valid_values), np.max(valid_values)
        data_range = data_max - data_min

        # 确保包含重要阈值线
        y_min = min(data_min - data_range * 0.15, THRESHOLD_LOW * 0.85)
        y_max = max(data_max + data_range * 0.15, THRESHOLD_NOISE * 1.05)

        # 如果数据范围太小，扩展显示范围
        if data_range < 500:
            center = (data_min + data_max) / 2
            y_min = center - 1000
            y_max = center + 1000

        ax1.set_ylim(max(0, y_min), y_max)

    # X轴刻度 - 修改为使用文件名
    if n_files <= 20:
        ax1.set_xticks(x)
        ax1.set_xticklabels(file_names, fontsize=9, rotation=45, ha='right')
    else:
        step = max(1, n_files // 20)
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels([file_names[i] for i in x[::step]], fontsize=9, rotation=45, ha='right')

    ax1.set_xlim(-0.5, n_files + 3)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # ==================== 图2: 热力图 ====================
    ax2 = fig.add_subplot(gs[1, :])

    # 统一时间轴长度
    max_len = max((len(s) for s in rolloff_series if len(s) > 0), default=100)
    target_len = min(max_len, 500)  # 限制最大长度

    rolloff_matrix = np.full((n_files, target_len), np.nan)
    for i, series in enumerate(rolloff_series):
        if len(series) > 0:
            # 重采样到统一长度
            indices = np.linspace(0, len(series) - 1, target_len).astype(int)
            rolloff_matrix[i] = series[indices]

    # 绘制热力图
    im = ax2.imshow(rolloff_matrix, aspect='auto', cmap='RdYlGn_r',
                    vmin=THRESHOLD_LOW * 0.8, vmax=THRESHOLD_NOISE * 1.1,
                    interpolation='nearest')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2, pad=0.02, shrink=0.9)
    cbar.set_label('频谱滚降点 (Hz)', fontsize=10)

    # 在颜色条上标注阈值
    cbar.ax.axhline(y=THRESHOLD_LOW, color='white', linewidth=2, linestyle='--')
    cbar.ax.axhline(y=THRESHOLD_NORMAL_HIGH, color='white', linewidth=2, linestyle='--')
    cbar.ax.axhline(y=THRESHOLD_NOISE, color='white', linewidth=2, linestyle='--')

    ax2.set_xlabel('时间进度 (%)', fontsize=11)
    ax2.set_ylabel('文件名称', fontsize=11)  # 修改Y轴标签
    ax2.set_title('频谱滚降点时序热力图（查看稳定性）', fontsize=13, fontweight='bold', pad=10)

    # 设置X轴为百分比
    x_ticks = np.linspace(0, target_len - 1, 11)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f'{int(p)}%' for p in np.linspace(0, 100, 11)], fontsize=9)

    # Y轴刻度 - 修改为使用文件名
    if n_files <= 30:
        ax2.set_yticks(range(n_files))
        ax2.set_yticklabels(file_names, fontsize=8)
    else:
        step = max(1, n_files // 20)
        ax2.set_yticks(range(0, n_files, step))
        ax2.set_yticklabels([file_names[i] for i in range(0, n_files, step)], fontsize=8)

    # ==================== 图3: 统计分布 ====================
    ax3 = fig.add_subplot(gs[2, 0])

    valid_rolloffs = mean_rolloffs[valid_mask]
    if len(valid_rolloffs) > 0:
        # 箱线图
        bp = ax3.boxplot(valid_rolloffs, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))

        # 叠加散点图
        jitter = np.random.normal(1, 0.04, len(valid_rolloffs))
        scatter_colors = [get_quality_color(v) for v in valid_rolloffs]
        ax3.scatter(jitter, valid_rolloffs, c=scatter_colors,
                    alpha=0.6, s=40, edgecolors='black', linewidths=0.5, zorder=3)

        # 添加阈值线
        ax3.axhline(y=THRESHOLD_LOW, color='#E74C3C', linestyle='--', linewidth=1.5)
        ax3.axhline(y=THRESHOLD_NORMAL_HIGH, color='#27AE60', linestyle='--', linewidth=1.5)
        ax3.axhline(y=THRESHOLD_NOISE, color='#E67E22', linestyle='--', linewidth=1.5)

        ax3.axhspan(THRESHOLD_LOW, THRESHOLD_NORMAL_HIGH, alpha=0.1, color='green')

    ax3.set_ylabel('频谱滚降点 (Hz)', fontsize=10)
    ax3.set_title('数值分布统计', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(['所有文件'])
    ax3.grid(True, alpha=0.3, axis='y')

    # ==================== 图4: 指标说明 ====================
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    # 统计信息
    n_low = np.sum(mean_rolloffs[valid_mask] < THRESHOLD_LOW)
    n_normal = np.sum((mean_rolloffs[valid_mask] >= THRESHOLD_LOW) &
                      (mean_rolloffs[valid_mask] <= THRESHOLD_NORMAL_HIGH))
    n_high = np.sum((mean_rolloffs[valid_mask] > THRESHOLD_NORMAL_HIGH) &
                    (mean_rolloffs[valid_mask] <= THRESHOLD_NOISE))
    n_noise = np.sum(mean_rolloffs[valid_mask] > THRESHOLD_NOISE)

    # 构建说明文本
    description = f"""
【频谱滚降点 (Spectral Roll-off)】

▸ 定义：累积能量达到 {int(roll_percent * 100)}% 的频率点
▸ 作用：检测音频高频完整性，识别人工处理痕迹

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
阈值判定标准：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔴 < {THRESHOLD_LOW} Hz    → 高频严重损失
  🟢 {THRESHOLD_LOW}-{THRESHOLD_NORMAL_HIGH} Hz  → 正常范围（最佳）
  🟡 {THRESHOLD_NORMAL_HIGH}-{THRESHOLD_NOISE} Hz → 偏高（可接受）
  🟠 > {THRESHOLD_NOISE} Hz   → 高频噪声异常

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本次分析统计：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  文件总数: {n_files}
  有效文件: {np.sum(valid_mask)}

  🔴 高频损失: {n_low} ({n_low / np.sum(valid_mask) * 100:.1f}%)
  🟢 正常范围: {n_normal} ({n_normal / np.sum(valid_mask) * 100:.1f}%)
  🟡 偏高: {n_high} ({n_high / np.sum(valid_mask) * 100:.1f}%)
  🟠 噪声异常: {n_noise} ({n_noise / np.sum(valid_mask) * 100:.1f}%)

  平均值: {np.nanmean(mean_rolloffs):.1f} Hz
  标准差: {np.nanstd(mean_rolloffs):.1f} Hz
  最小值: {np.nanmin(mean_rolloffs):.1f} Hz
  最大值: {np.nanmax(mean_rolloffs):.1f} Hz

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 解读提示：
• 数值稳定在3000-8000Hz为最佳
• 随训练轮数应趋于稳定
• 热力图横向条纹应均匀
"""

    ax4.text(0.05, 0.95, description, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='sans-serif',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA',
                       edgecolor='#DEE2E6', alpha=0.95))

    # ==================== 图例 ====================
    legend_elements = [
        Patch(facecolor='#27AE60', alpha=0.7, edgecolor='black', label='正常 (3000-8000 Hz)'),
        Patch(facecolor='#F39C12', alpha=0.7, edgecolor='black', label='偏高 (8000-10000 Hz)'),
        Patch(facecolor='#E67E22', alpha=0.7, edgecolor='black', label='噪声 (>10000 Hz)'),
        Patch(facecolor='#E74C3C', alpha=0.7, edgecolor='black', label='高频损失 (<3000 Hz)'),
        Line2D([0], [0], color='blue', linewidth=2, label='趋势线'),
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.04), ncol=5, fontsize=10,
               framealpha=0.9)

    # ==================== 显示图表 ====================
    plt.show()

    # ==================== 返回结果 ====================
    return {
        'file_names': file_names,
        'mean_rolloffs': mean_rolloffs.tolist(),
        'std_rolloffs': std_rolloffs.tolist(),
        'durations': durations.tolist(),
        'statistics': {
            'mean': float(np.nanmean(mean_rolloffs)),
            'std': float(np.nanstd(mean_rolloffs)),
            'min': float(np.nanmin(mean_rolloffs)),
            'max': float(np.nanmax(mean_rolloffs)),
            'n_low': int(n_low),
            'n_normal': int(n_normal),
            'n_high': int(n_high),
            'n_noise': int(n_noise)
        }
    }


# ==================== 使用示例 ====================
if __name__ == '__main__':
    import glob

    # 示例：获取文件夹中所有wav文件
    # wav_files = sorted(glob.glob(r"D:\your_path\*.wav"))

    # 或者直接指定文件列表
    wav_files = [
        r"path/to/model_epoch_100.wav",
        r"path/to/model_epoch_200.wav",
        r"path/to/model_epoch_300.wav",
        # ... 更多文件
    ]

    # 调用分析函数
    results = analyze_spectral_rolloff(wav_files)

    # 打印统计结果
    print("\n分析完成！统计摘要：")
    print(f"  平均滚降点: {results['statistics']['mean']:.1f} Hz")
    print(f"  正常范围文件数: {results['statistics']['n_normal']}")