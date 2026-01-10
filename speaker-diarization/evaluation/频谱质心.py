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

+ 频谱质心（Spectral Centroid）
    + 含义：频谱“重心”，公式：Centroid = Σ(f_i × A_i) / Σ A_i；反映“亮度”，AI过亮→金属感，过暗→闷沉。
    + 1000–3000 Hz：正常
    + 小于 1000 Hz：闷
    + 大于 4000 Hz：刺 / 金属感
"""

# pip install numpy librosa matplotlib soundfile

"""
AI翻唱质量评价 - 频谱质心 (Spectral Centroid) 分析

依赖安装:
    pip install numpy librosa matplotlib soundfile

使用方法:
    from spectral_centroid_analyzer import analyze_spectral_centroid
    analyze_spectral_centroid(["path1.wav", "path2.wav", ...])
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
import warnings


def analyze_spectral_centroid(wav_paths: List[str], max_workers: int = 16) -> Optional[dict]:
    """
    分析多个WAV文件的频谱质心并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数排序）
        max_workers: 并发处理的最大线程数

    返回:
        dict: 包含所有计算结果的字典，如果失败返回None
    """

    # ========== 导入依赖（函数内部导入，便于错误提示） ==========
    try:
        import librosa
    except ImportError:
        print("错误: 请先安装 librosa: pip install librosa")
        return None

    warnings.filterwarnings('ignore')

    # ========== 单文件处理函数 ==========
    def compute_single_file(wav_path: str) -> Tuple[str, float, float, float, np.ndarray]:
        """
        计算单个文件的频谱质心

        返回: (文件名, 平均值, 标准差, 中位数, 时间序列)
        """
        try:
            # 加载音频
            y, sr = librosa.load(wav_path, sr=None)

            # 计算频谱质心 (每帧一个值)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

            filename = Path(wav_path).stem
            return (
                filename,
                float(np.mean(centroid)),
                float(np.std(centroid)),
                float(np.median(centroid)),
                centroid
            )
        except Exception as e:
            filename = Path(wav_path).stem
            print(f"  ⚠ 处理文件 '{filename}' 时出错: {e}")
            return (filename, np.nan, np.nan, np.nan, np.array([]))

    # ========== 并发处理所有文件 ==========
    n_files = len(wav_paths)
    if n_files == 0:
        print("错误: 文件列表为空")
        return None

    print(f"📂 正在处理 {n_files} 个文件...")

    results = [None] * n_files
    completed = 0

    with ThreadPoolExecutor(max_workers=min(max_workers, n_files)) as executor:
        future_to_idx = {
            executor.submit(compute_single_file, path): idx
            for idx, path in enumerate(wav_paths)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            # 进度显示
            if completed % 10 == 0 or completed == n_files:
                print(f"  进度: {completed}/{n_files} ({100 * completed // n_files}%)")

    print("✅ 文件处理完成，正在生成图表...\n")

    # ========== 提取数据 ==========
    filenames = [r[0] for r in results]
    means = np.array([r[1] for r in results])
    stds = np.array([r[2] for r in results])
    medians = np.array([r[3] for r in results])

    valid_mask = ~np.isnan(means)
    valid_count = np.sum(valid_mask)

    if valid_count == 0:
        print("错误: 所有文件处理都失败了")
        return None

    # ========== 配置 matplotlib ==========
    # 设置常规字体（非等宽字体）
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'Hiragino Sans GB',
                                   'WenQuanYi Micro Hei', 'PingFang SC',
                                   'Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # ========== 根据文件数量自适应图表尺寸 ==========
    if n_files <= 30:
        fig_width = max(14, n_files * 0.45)
    elif n_files <= 60:
        fig_width = max(16, n_files * 0.35)
    else:
        fig_width = max(20, min(32, n_files * 0.28))
    fig_height = 14

    # ========== 创建图表布局 ==========
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 使用 GridSpec 灵活布局
    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[5, 2.5, 0.5],
        width_ratios=[4, 1, 1, 1.2],
        hspace=0.25,
        wspace=0.2,
        left=0.06, right=0.98, top=0.93, bottom=0.05
    )

    ax_main = fig.add_subplot(gs[0, :3])  # 主图：趋势线
    ax_info = fig.add_subplot(gs[0, 3])  # 右侧：指标说明
    ax_deviation = fig.add_subplot(gs[1, :3])  # 偏离度图
    ax_stats = fig.add_subplot(gs[1, 3])  # 右侧：统计信息

    x = np.arange(n_files)

    # ========== 颜色定义 ==========
    COLOR_NORMAL = '#27ae60'  # 绿色 - 正常
    COLOR_SLIGHTLY_BRIGHT = '#f39c12'  # 橙色 - 略亮
    COLOR_DULL = '#3498db'  # 蓝色 - 闷
    COLOR_HARSH = '#e74c3c'  # 红色 - 刺
    COLOR_GRAY = '#95a5a6'  # 灰色 - 无效
    COLOR_LINE = '#2c3e50'  # 深色线条

    # 根据数值分配颜色
    def get_color(value):
        if np.isnan(value):
            return COLOR_GRAY
        elif value < 1000:
            return COLOR_DULL
        elif value > 4000:
            return COLOR_HARSH
        elif value > 3000:
            return COLOR_SLIGHTLY_BRIGHT
        else:
            return COLOR_NORMAL

    point_colors = [get_color(v) for v in means]

    # ========== 主图：频谱质心趋势 ==========

    # 绘制趋势线
    ax_main.plot(x, means, color=COLOR_LINE, linewidth=1.5, alpha=0.5, zorder=1)

    # 绘制标准差范围
    if valid_count > 1:
        ax_main.fill_between(
            x[valid_mask],
            (means - stds)[valid_mask],
            (means + stds)[valid_mask],
            alpha=0.12, color='#3498db', label='±1 标准差范围'
        )

    # 绘制散点（彩色标记）
    scatter = ax_main.scatter(
        x, means,
        c=point_colors,
        s=70,
        edgecolors='white',
        linewidth=1,
        zorder=3
    )

    # 阈值区域填充
    y_plot_min = max(0, np.nanmin(means) - 800)
    y_plot_max = np.nanmax(means) + 800

    ax_main.axhspan(1000, 3000, alpha=0.06, color=COLOR_NORMAL, zorder=0)
    ax_main.axhspan(y_plot_min, 1000, alpha=0.04, color=COLOR_DULL, zorder=0)
    ax_main.axhspan(4000, y_plot_max, alpha=0.04, color=COLOR_HARSH, zorder=0)

    # 阈值参考线
    ax_main.axhline(y=1000, color=COLOR_DULL, linestyle='--', linewidth=2,
                    alpha=0.9, label='1000 Hz: 低于此值偏闷')
    ax_main.axhline(y=3000, color=COLOR_SLIGHTLY_BRIGHT, linestyle='--', linewidth=2,
                    alpha=0.9, label='3000 Hz: 正常范围上限')
    ax_main.axhline(y=4000, color=COLOR_HARSH, linestyle='--', linewidth=2,
                    alpha=0.9, label='4000 Hz: 高于此值偏刺')

    # Y轴范围优化（放大差异）
    valid_means = means[valid_mask]
    data_min, data_max = np.nanmin(valid_means), np.nanmax(valid_means)
    data_range = data_max - data_min

    if data_range < 200:
        # 差异很小时，放大显示
        center = np.nanmean(valid_means)
        y_min = center - 350
        y_max = center + 350
    elif data_range < 500:
        margin = 200
        y_min = data_min - margin
        y_max = data_max + margin
    else:
        margin = data_range * 0.12
        y_min = data_min - margin
        y_max = data_max + margin

    # 确保阈值线可见
    y_min = min(y_min, 700)
    y_max = max(y_max, 4300)
    ax_main.set_ylim(y_min, y_max)

    # 标签和标题
    ax_main.set_ylabel('频谱质心 (Hz)', fontsize=12, fontweight='bold')
    ax_main.set_title(
        'AI翻唱质量评价 — 频谱质心 (Spectral Centroid) 趋势分析',
        fontsize=14, fontweight='bold', pad=12
    )

    # X轴标签（自适应）
    def get_tick_positions(n, max_ticks=25):
        if n <= max_ticks:
            return list(range(n))
        step = n // max_ticks + 1
        ticks = list(range(0, n, step))
        if (n - 1) not in ticks:
            ticks.append(n - 1)
        return ticks

    tick_positions = get_tick_positions(n_files)
    ax_main.set_xticks(tick_positions)

    if n_files <= 25:
        ax_main.set_xticklabels(
            [filenames[i] for i in tick_positions],
            rotation=50, ha='right', fontsize=8
        )
    else:
        ax_main.set_xticklabels(
            [filenames[i] for i in tick_positions],
            rotation=55, ha='right', fontsize=7
        )

    ax_main.grid(True, alpha=0.35, linestyle='-', linewidth=0.5)
    ax_main.legend(loc='upper left', fontsize=9, framealpha=0.92)

    # 标注最值点
    if valid_count >= 2:
        max_idx = np.nanargmax(means)
        min_idx = np.nanargmin(means)

        # 最高点
        ax_main.annotate(
            f'最高: {means[max_idx]:.0f} Hz\n({filenames[max_idx]})',
            xy=(max_idx, means[max_idx]),
            xytext=(15, 15),
            textcoords='offset points',
            fontsize=8,
            color=COLOR_HARSH,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_HARSH, lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0)  # 透明背景
        )

        # 最低点
        ax_main.annotate(
            f'最低: {means[min_idx]:.0f} Hz\n({filenames[min_idx]})',
            xy=(min_idx, means[min_idx]),
            xytext=(15, -25),
            textcoords='offset points',
            fontsize=8,
            color=COLOR_DULL,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_DULL, lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0)  # 透明背景
        )

    # ========== 右上角：指标说明 ==========
    ax_info.axis('off')

    info_text = """频谱质心
(Spectral Centroid)
━━━━━━━━━━━━━━━━━━━━━

📖 含义
  频谱的"重心"位置
  反映声音的"亮度"特征

📐 计算公式
  Centroid = Σ(f × A) / Σ A
  (频率加权平均)

━━━━━━━━━━━━━━━━━━━━━

✅ 质量阈值参考

  🟢 1000 - 3000 Hz
     正常范围

  🔵 < 1000 Hz
     偏闷沉

  🟠 3000 - 4000 Hz
     略亮（可接受）

  🔴 > 4000 Hz
     偏刺 / 金属感

━━━━━━━━━━━━━━━━━━━━━

⚠️ AI翻唱常见问题

  • 过亮 → 金属感、不自然
  • 过暗 → 闷沉、缺乏活力
"""

    ax_info.text(
        0.05, 0.98, info_text,
        transform=ax_info.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        linespacing=1.3,
        family='sans-serif'
        # 不设置 bbox，背景完全透明
    )

    # ========== 偏离度图 ==========
    optimal_center = 2000  # 理想中心值
    deviations = means - optimal_center

    # 偏离度颜色
    def get_deviation_color(dev):
        if np.isnan(dev):
            return COLOR_GRAY
        elif dev > 2000:
            return '#c0392b'  # 深红
        elif dev > 1000:
            return COLOR_HARSH
        elif dev < -1000:
            return COLOR_DULL
        elif abs(dev) > 500:
            return COLOR_SLIGHTLY_BRIGHT
        else:
            return COLOR_NORMAL

    bar_colors = [get_deviation_color(d) for d in deviations]

    # 绘制柱状图
    bars = ax_deviation.bar(x, deviations, color=bar_colors, alpha=0.8, width=0.85)

    # 参考线
    ax_deviation.axhline(y=0, color=COLOR_LINE, linestyle='-', linewidth=2)
    ax_deviation.axhline(y=1000, color=COLOR_SLIGHTLY_BRIGHT, linestyle=':', linewidth=1.5, alpha=0.7)
    ax_deviation.axhline(y=-1000, color=COLOR_SLIGHTLY_BRIGHT, linestyle=':', linewidth=1.5, alpha=0.7)
    ax_deviation.axhspan(-1000, 1000, alpha=0.06, color=COLOR_NORMAL)

    ax_deviation.set_xlabel('文件序号（按模型训练轮数递增 →）', fontsize=11, fontweight='bold')
    ax_deviation.set_ylabel('偏离度 (Hz)', fontsize=11, fontweight='bold')
    ax_deviation.set_title(
        '与理想值 (2000 Hz) 的偏离程度   |   绿色区域(-1000~+1000)为良好范围',
        fontsize=11, fontweight='bold'
    )

    ax_deviation.set_xticks(tick_positions)
    ax_deviation.set_xticklabels([str(i) for i in tick_positions], fontsize=8)
    ax_deviation.grid(True, alpha=0.3, axis='y')

    # ========== 右下角：统计信息 ==========
    ax_stats.axis('off')

    # 计算质量分布
    normal_count = np.sum((means >= 1000) & (means <= 3000))
    dull_count = np.sum(means < 1000)
    harsh_count = np.sum(means > 4000)
    slightly_bright_count = np.sum((means > 3000) & (means <= 4000))

    # 趋势判断
    if valid_count >= 3:
        first_third = np.nanmean(means[:n_files // 3])
        last_third = np.nanmean(means[-n_files // 3:])
        trend_diff = last_third - first_third
        if trend_diff > 100:
            trend_str = "📈 上升趋势（变亮）"
        elif trend_diff < -100:
            trend_str = "📉 下降趋势（变暗）"
        else:
            trend_str = "➡️ 基本平稳"
    else:
        trend_str = "—"

    stats_text = f"""📊 统计摘要
━━━━━━━━━━━━━━━━━━━━━

  文件总数:  {n_files}
  有效文件:  {valid_count}

━━━━━━━━━━━━━━━━━━━━━

  平均值:    {np.nanmean(means):.1f} Hz
  中位数:    {np.nanmedian(means):.1f} Hz
  标准差:    {np.nanstd(means):.1f} Hz

  最小值:    {np.nanmin(means):.1f} Hz
  最大值:    {np.nanmax(means):.1f} Hz
  极差:      {data_range:.1f} Hz

━━━━━━━━━━━━━━━━━━━━━

📊 质量分布

  🟢 正常:   {normal_count} ({100 * normal_count // n_files}%)
  🔵 偏闷:   {dull_count} ({100 * dull_count // n_files}%)
  🟠 略亮:   {slightly_bright_count} ({100 * slightly_bright_count // n_files}%)
  🔴 偏刺:   {harsh_count} ({100 * harsh_count // n_files}%)

━━━━━━━━━━━━━━━━━━━━━

📈 趋势分析

  {trend_str}
"""

    ax_stats.text(
        0.05, 0.98, stats_text,
        transform=ax_stats.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        linespacing=1.25,
        family='sans-serif'
    )

    # ========== 底部总体评价 ==========
    mean_val = np.nanmean(means)

    if 1000 <= mean_val <= 3000:
        if normal_count >= n_files * 0.8:
            quality_str = "✅ 整体质量优秀：频谱质心稳定在正常范围内"
            quality_color = COLOR_NORMAL
        else:
            quality_str = "✓ 整体质量良好：平均值在正常范围，但存在波动"
            quality_color = COLOR_NORMAL
    elif mean_val < 1000:
        quality_str = "⚠️ 整体偏闷沉：建议调整模型参数增加亮度"
        quality_color = COLOR_DULL
    elif mean_val > 4000:
        quality_str = "⚠️ 整体偏刺耳：建议调整模型参数降低亮度"
        quality_color = COLOR_HARSH
    else:
        quality_str = "○ 整体略偏亮：在可接受范围，可根据需要微调"
        quality_color = COLOR_SLIGHTLY_BRIGHT

    fig.text(
        0.5, 0.01, quality_str,
        ha='center', fontsize=13,
        fontweight='bold', color=quality_color
    )

    # ========== 显示图表 ==========
    plt.show()

    print("📊 图表已显示")

    # ========== 返回计算结果 ==========
    return {
        'filenames': filenames,
        'means': means.tolist(),
        'stds': stds.tolist(),
        'medians': medians.tolist(),
        'statistics': {
            'mean': float(np.nanmean(means)),
            'median': float(np.nanmedian(means)),
            'std': float(np.nanstd(means)),
            'min': float(np.nanmin(means)),
            'max': float(np.nanmax(means)),
            'range': float(data_range)
        }
    }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import glob

    # 示例：获取某目录下所有 wav 文件
    # wav_files = sorted(glob.glob("path/to/your/wav/files/*.wav"))

    # 示例调用
    wav_files = [
        "model_epoch_100.wav",
        "model_epoch_200.wav",
        "model_epoch_300.wav",
        # ... 更多文件
    ]

    # 调用分析函数
    # results = analyze_spectral_centroid(wav_files)