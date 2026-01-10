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

+ 频谱平坦度（Spectral Flatness）
    + 含义：频谱接近噪声还是谐波，公式：SF = 几何平均(功率谱) / 算术平均(功率谱)；高值表示噪声感强。
    + 接近 0：谐波型（好）
    + 0.1–0.3：有气声
    + 接近 1：噪声化
"""

# pip install numpy librosa matplotlib soundfile

"""
AI翻唱质量评估 - 频谱平坦度 (Spectral Flatness) 分析工具

依赖安装:
    pip install numpy librosa matplotlib soundfile

作者: AI Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
import warnings


def analyze_spectral_flatness(wav_paths: List[str]) -> dict:
    """
    分析多个WAV文件的频谱平坦度并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数排序）

    返回:
        dict: 包含各文件分析结果的字典
    """

    # ==================== 延迟导入 ====================
    import librosa
    import matplotlib.font_manager as fm

    # ==================== 字体配置（常规字体，非等宽） ====================
    chinese_fonts = [
        'Microsoft YaHei', 'SimHei', 'PingFang SC',
        'Hiragino Sans GB', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'STHeiti'
    ]
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    font_found = False
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = [font, 'sans-serif']
            font_found = True
            break

    if not font_found:
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        warnings.warn("未找到中文字体，部分中文可能显示异常。建议安装 Microsoft YaHei 或 SimHei 字体。")

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10

    # ==================== 单文件处理函数 ====================
    def process_single_file(args: Tuple[int, str]) -> Optional[Tuple]:
        """处理单个WAV文件，计算频谱平坦度"""
        idx, wav_path = args
        try:
            # 加载音频文件
            y, sr = librosa.load(wav_path, sr=None, mono=True)

            # 计算频谱平坦度 (返回每帧的值)
            # SF = 几何平均(功率谱) / 算术平均(功率谱)
            sf = librosa.feature.spectral_flatness(y=y)[0]

            return (
                idx,
                Path(wav_path).stem,  # 文件名（不含扩展名）
                float(np.mean(sf)),  # 均值
                float(np.std(sf)),  # 标准差
                float(np.min(sf)),  # 最小值
                float(np.max(sf)),  # 最大值
                float(np.median(sf)),  # 中位数
            )
        except Exception as e:
            print(f"[错误] 处理失败: {wav_path}\n        原因: {e}")
            return None

    # ==================== 并发处理所有文件 ====================
    n_files = len(wav_paths)
    print(f"🎵 开始分析 {n_files} 个WAV文件的频谱平坦度...")

    results = []
    max_workers = min(8, max(1, n_files))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(process_single_file, (i, path))
            for i, path in enumerate(wav_paths)
        ]

        # 收集结果并显示进度
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == n_files:
                print(f"   进度: {completed}/{n_files}")

    if not results:
        print("❌ 没有成功处理任何文件！请检查文件路径是否正确。")
        return {}

    # 按原始顺序排序
    results.sort(key=lambda x: x[0])

    # 解包数据
    indices, names, means, stds, mins, maxs, medians = zip(*results)
    names = list(names)
    means = np.array(means)
    stds = np.array(stds)
    mins = np.array(mins)
    maxs = np.array(maxs)

    n_valid = len(names)
    x = np.arange(n_valid)

    print(f"✅ 成功处理 {n_valid}/{n_files} 个文件\n")

    # ==================== 创建图表 ====================
    # 动态计算图表尺寸
    fig_width = max(14, min(48, n_valid * 0.4))
    fig_height = 14

    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={'height_ratios': [1.15, 1]}
    )

    fig.suptitle(
        'AI翻唱质量评估报告 - 频谱平坦度 (Spectral Flatness)',
        fontsize=16, fontweight='bold', y=0.995
    )

    # ========== 子图1: 趋势折线图 ==========
    ax1 = axes[0]

    # 绘制标准差范围带
    ax1.fill_between(
        x, means - stds, means + stds,
        alpha=0.2, color='#3498db', label='±1σ 标准差范围'
    )

    # 绘制最大-最小值范围（更浅）
    ax1.fill_between(
        x, mins, maxs,
        alpha=0.08, color='#9b59b6', label='最小-最大值范围'
    )

    # 主数据折线
    marker_size = max(3, 10 - n_valid // 12)
    ax1.plot(
        x, means, 'o-',
        color='#2980b9', linewidth=2.2,
        markersize=marker_size,
        markerfacecolor='white',
        markeredgewidth=1.5,
        markeredgecolor='#2980b9',
        label='频谱平坦度均值', zorder=5
    )

    # ===== Y轴范围优化：智能自适应，放大差异 =====
    data_min, data_max = means.min(), means.max()
    data_range = data_max - data_min

    # 计算数据的中心点
    data_center = (data_min + data_max) / 2

    # 确定Y轴范围（根据数据范围和差异大小动态调整）
    if data_range == 0:
        # 所有数据相同的情况
        y_padding = 0.01  # 固定的小间距
        y_lower = max(0, data_min - y_padding)
        y_upper = min(1.0, data_max + y_padding)
    elif data_range < 0.001:
        # 差异极小的情况：扩大显示范围
        y_padding = data_range * 20  # 放大20倍
        y_lower = max(0, data_center - y_padding)
        y_upper = min(1.0, data_center + y_padding)
    elif data_range < 0.01:
        # 差异较小的情况：适度放大
        y_padding = data_range * 5  # 放大5倍
        y_lower = max(0, data_center - y_padding)
        y_upper = min(1.0, data_center + y_padding)
    elif data_range < 0.05:
        # 差异一般的情况：适度放大
        y_padding = data_range * 2  # 放大2倍
        y_lower = max(0, data_center - y_padding)
        y_upper = min(1.0, data_center + y_padding)
    else:
        # 差异足够大的情况：使用正常范围
        y_padding = data_range * 0.3  # 30%的边距
        y_lower = max(0, data_min - y_padding)
        y_upper = min(1.0, data_max + y_padding)

    # 确保范围有效且不为零
    if y_upper - y_lower < 1e-10:
        y_lower = max(0, data_center - 0.01)
        y_upper = min(1.0, data_center + 0.01)

    # 应用Y轴范围
    ax1.set_ylim(y_lower, y_upper)

    ax1.set_xlabel('模型训练轮数 (按顺序递增) →', fontsize=11, fontweight='medium')
    ax1.set_ylabel('频谱平坦度值', fontsize=11, fontweight='medium')
    ax1.set_title('📈 训练过程中频谱平坦度变化趋势（值越低 = 音质越好）', fontsize=12, pad=10)

    # X轴标签优化
    max_labels = 25
    if n_valid <= max_labels:
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=50, ha='right', fontsize=8)
    else:
        step = (n_valid - 1) // (max_labels - 1)
        tick_positions = list(range(0, n_valid, step))
        if (n_valid - 1) not in tick_positions:
            tick_positions.append(n_valid - 1)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(
            [names[i] for i in tick_positions],
            rotation=50, ha='right', fontsize=8
        )

    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95,
               edgecolor='#bdc3c7', fancybox=True)
    ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-0.5, n_valid - 0.5)

    # 指标说明文本框（透明背景）
    desc_text = (
        "【频谱平坦度 Spectral Flatness】\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "计算公式:\n"
        "  SF = 几何平均(功率谱) ÷ 算术平均(功率谱)\n\n"
        "质量评判标准:\n"
        "  • ≈ 0      谐波结构清晰，人声质量优秀\n"
        "  • 0.1~0.3  存在气声或轻微背景噪声\n"
        "  • → 1      趋近白噪声，音质较差\n\n"
        "▶ 数值越低越好 | 下降趋势 = 模型在改进"
    )
    ax1.text(
        0.012, 0.97, desc_text,
        transform=ax1.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='none',  # 透明背景
            edgecolor='#95a5a6',
            linewidth=1
        ),
        linespacing=1.4,
        family='sans-serif'
    )

    # ========== 子图2: 柱状对比图 ==========
    ax2 = axes[1]

    # 根据阈值为每个柱子着色
    colors = []
    for m in means:
        if m < 0.1:
            colors.append('#27ae60')  # 绿色 - 优秀
        elif m < 0.3:
            colors.append('#f39c12')  # 橙色 - 一般
        else:
            colors.append('#e74c3c')  # 红色 - 较差

    # 绘制柱状图
    bars = ax2.bar(
        x, means,
        width=0.8,
        color=colors,
        alpha=0.78,
        edgecolor='#2c3e50',
        linewidth=0.5
    )

    # 误差线（标准差）
    cap_size = max(1, 5 - n_valid // 20)
    ax2.errorbar(
        x, means, yerr=stds,
        fmt='none', ecolor='#7f8c8d',
        capsize=cap_size, alpha=0.6, linewidth=1
    )

    # 不强制显示阈值线，以免挤压数据区间
    # 只有在数据范围包含阈值附近时才显示
    if y_lower <= 0.3 <= y_upper:
        ax2.axhline(y=0.3, color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.6)

    if y_lower <= 0.1 <= y_upper:
        ax2.axhline(y=0.1, color='#f39c12', linestyle='--', linewidth=1.2, alpha=0.6)

    # 使用与子图1相同的Y轴范围，确保视图一致性
    ax2.set_ylim(y_lower, y_upper)
    ax2.set_xlabel('文件名称（按模型轮数排序）', fontsize=11, fontweight='medium')
    ax2.set_ylabel('频谱平坦度值', fontsize=11, fontweight='medium')
    ax2.set_title(
        '📊 各文件频谱平坦度对比  [ 🟢 <0.1 优秀 | 🟡 0.1~0.3 一般 | 🔴 >0.3 较差 ]',
        fontsize=12, pad=10
    )

    # X轴标签
    max_bar_labels = 35
    if n_valid <= max_bar_labels:
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=55, ha='right', fontsize=7)
    else:
        step = (n_valid - 1) // (max_bar_labels - 1)
        tick_positions = list(range(0, n_valid, step))
        if (n_valid - 1) not in tick_positions:
            tick_positions.append(n_valid - 1)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(
            [names[i] for i in tick_positions],
            rotation=55, ha='right', fontsize=7
        )

    ax2.grid(True, alpha=0.4, axis='y', linestyle='-', linewidth=0.5)
    ax2.set_xlim(-0.5, n_valid - 0.5)

    # ========== 底部统计摘要 ==========
    best_idx = int(np.argmin(means))
    worst_idx = int(np.argmax(means))

    # 判断趋势
    if n_valid >= 3:
        first_third = means[:n_valid // 3].mean()
        last_third = means[-n_valid // 3:].mean()
        if last_third < first_third * 0.95:
            trend_text = "📉 下降趋势 (模型改进中)"
            trend_color = '#27ae60'
        elif last_third > first_third * 1.05:
            trend_text = "📈 上升趋势 (模型退化)"
            trend_color = '#e74c3c'
        else:
            trend_text = "➡️ 平稳趋势"
            trend_color = '#3498db'
    else:
        trend_text = "数据点不足"
        trend_color = '#7f8c8d'

    stats_text = (
        f"📋 统计摘要:   "
        f"平均值 = {means.mean():.4f}   |   "
        f"最佳 = {means.min():.4f} 【{names[best_idx]}】   |   "
        f"最差 = {means.max():.4f} 【{names[worst_idx]}】   |   "
        f"趋势: {trend_text}"
    )
    fig.text(
        0.5, 0.008, stats_text,
        ha='center', fontsize=10.5,
        color='#2c3e50',
        fontweight='medium',
        family='sans-serif'
    )

    # ========== 调整布局并显示 ==========
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(hspace=0.38)

    print("📊 正在显示分析图表...")
    plt.show()

    # ========== 返回分析结果 ==========
    return {
        'file_names': names,
        'means': means.tolist(),
        'stds': stds.tolist(),
        'mins': mins.tolist(),
        'maxs': maxs.tolist(),
        'best_file': names[best_idx],
        'best_value': float(means.min()),
        'worst_file': names[worst_idx],
        'worst_value': float(means.max()),
        'overall_mean': float(means.mean()),
    }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import glob
    import os

    # 示例：指定目录获取所有WAV文件
    audio_dir = "./audio_outputs"  # 修改为你的目录

    # 获取并排序WAV文件
    wav_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))

    if wav_files:
        print(f"找到 {len(wav_files)} 个WAV文件")
        results = analyze_spectral_flatness(wav_files)

        # 打印最佳结果
        if results:
            print(f"\n🏆 最佳文件: {results['best_file']}")
            print(f"   频谱平坦度: {results['best_value']:.4f}")
    else:
        print(f"在 {audio_dir} 目录下未找到WAV文件")
        print("\n使用示例:")
        print('  wav_files = ["path/to/file1.wav", "path/to/file2.wav", ...]')
        print('  analyze_spectral_flatness(wav_files)')