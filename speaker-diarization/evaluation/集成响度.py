"""
1. 我打算使用以下指标对ai翻唱的wav文件进行质量评价。
2. 我只有使用ai翻唱出来的多个wav文件，我能提供这些文件的路径。
3. 判断以下指标，只有wav文件路径，这些文件之间是否能对比出优劣，如果对比不出优劣就不需要再继续了
4. 如果能对比出优劣，写一个python函数，入参是wav文件路径的字符串数组
5. 该python函数实现以下指标的计算，并且将计算结果画为图表进行可视化对比
6. 图表的类型，需要根据指标的特殊进行选择，目的是能更加直观的看出各个wav文件的优劣
7. 图表的数轴标度，为了避免不同文件之间的指标差异过小，在图中看不出区别，需要更加明显的处理
8. 在图表中增加该指标的文字描述，阈值的辅助信息
9. 尽量将代码都收敛到函数内部，方便调用
10. 最后提供一个完整可用的python函数，以及其需要安装的依赖

+ 集成响度（Integrated Loudness, LUFS）
    + 含义：符合ITU-R BS.1770标准的整体感知响度，基于人耳听感（比RMS更准确），用于判断音频能量是否异常（如AI模型输出过弱或过强）。
    + -14 ~ -16：主流音乐推荐，平衡存在感
    + -18 ~ -20：偏小，可能缺乏冲击力
    + 大于-12：过度压缩或失真
    + 小于-22：能量不足，AI常见问题
    + AI翻唱建议目标：-16 ± 2 LUFS
"""

# pip install numpy matplotlib soundfile pyloudnorm

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pyloudnorm as pyln
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def analyze_integrated_loudness(wav_paths: List[str],
                                save_path: str = None,
                                show_plot: bool = True) -> Dict[str, float]:
    """
    分析多个WAV文件的集成响度(LUFS)并可视化对比

    Args:
        wav_paths: WAV文件路径的字符串数组
        save_path: 图表保存路径（可选）
        show_plot: 是否显示图表

    Returns:
        Dict[str, float]: 文件名到LUFS值的映射
    """

    # ========== 1. 计算每个文件的LUFS ==========
    results = []
    file_names = []
    errors = []

    for path in wav_paths:
        try:
            # 读取音频文件
            data, rate = sf.read(path)

            # 如果是单声道，转换为二维数组
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)

            # 创建符合 ITU-R BS.1770 标准的响度测量器
            meter = pyln.Meter(rate)

            # 计算集成响度
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loudness = meter.integrated_loudness(data)

            # 处理静音或极低音量的情况
            if np.isinf(loudness) or np.isnan(loudness):
                loudness = -70.0  # 设置一个极低值表示静音

            results.append(loudness)
            file_names.append(Path(path).stem)  # 使用不带扩展名的文件名

        except Exception as e:
            errors.append(f"{Path(path).name}: {str(e)}")
            continue

    if errors:
        print("⚠️ 处理以下文件时出错:")
        for err in errors:
            print(f"  - {err}")

    if not results:
        raise ValueError("没有成功处理任何文件")

    # ========== 2. 创建可视化图表 ==========
    fig, ax = plt.subplots(figsize=(14, max(8, len(file_names) * 0.6 + 3)))

    # ---------- 2.1 定义颜色（根据与理想值的偏差） ----------
    def get_color_and_status(lufs: float) -> Tuple[str, str]:
        """根据LUFS值返回颜色和状态"""
        if -18 <= lufs <= -14:
            return '#2ecc71', '优秀'  # 绿色
        elif -20 <= lufs < -18:
            return '#f39c12', '偏弱'  # 橙色
        elif -14 < lufs <= -12:
            return '#f39c12', '偏强'  # 橙色
        elif lufs > -12:
            return '#e74c3c', '过载'  # 红色
        elif lufs < -22:
            return '#e74c3c', '过弱'  # 红色
        else:
            return '#f39c12', '警告'  # 橙色

    colors = []
    statuses = []
    for lufs in results:
        color, status = get_color_and_status(lufs)
        colors.append(color)
        statuses.append(status)

    # ---------- 2.2 绘制水平条形图 ----------
    y_pos = np.arange(len(file_names))

    # 为了更直观显示，我们绘制相对于理想值-16的偏差
    # 但同时保持实际LUFS值的显示
    bars = ax.barh(y_pos, results, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    # ---------- 2.3 添加参考区间和阈值线 ----------
    # 理想范围背景
    ax.axvspan(-18, -14, alpha=0.15, color='green', zorder=0)

    # 阈值线
    ax.axvline(x=-16, color='#27ae60', linestyle='-', linewidth=2.5,
               label='理想值 (-16 LUFS)', zorder=5)
    ax.axvline(x=-14, color='#27ae60', linestyle='--', linewidth=1.5,
               label='推荐上限 (-14 LUFS)', alpha=0.7)
    ax.axvline(x=-18, color='#27ae60', linestyle='--', linewidth=1.5,
               label='推荐下限 (-18 LUFS)', alpha=0.7)
    ax.axvline(x=-12, color='#e74c3c', linestyle=':', linewidth=2,
               label='过载警告 (-12 LUFS)')
    ax.axvline(x=-22, color='#e74c3c', linestyle=':', linewidth=2,
               label='能量不足警告 (-22 LUFS)')

    # ---------- 2.4 在条形上显示数值和状态 ----------
    for i, (bar, val, status) in enumerate(zip(bars, results, statuses)):
        # 数值标签
        text_x = val + 0.3 if val < -16 else val - 0.3
        ha = 'left' if val < -16 else 'right'

        ax.text(text_x, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f} LUFS [{status}]',
                va='center', ha=ha, fontsize=10, fontweight='bold',
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

    # ---------- 2.5 优化坐标轴 ----------
    ax.set_yticks(y_pos)
    ax.set_yticklabels(file_names, fontsize=11)
    ax.set_xlabel('集成响度 (LUFS)', fontsize=12, fontweight='bold')
    ax.set_title('AI翻唱音频质量评估 - 集成响度(Integrated Loudness)分析',
                 fontsize=14, fontweight='bold', pad=20)

    # 动态调整X轴范围，确保差异可见
    data_min = min(results)
    data_max = max(results)

    # 确保显示所有参考线
    plot_min = min(data_min, -24) - 2
    plot_max = max(data_max, -10) + 2

    # 如果数据范围太小，扩展显示范围以突出差异
    data_range = data_max - data_min
    if data_range < 4:
        center = (data_max + data_min) / 2
        plot_min = min(plot_min, center - 5)
        plot_max = max(plot_max, center + 5)

    ax.set_xlim(plot_min, plot_max)

    # 网格线
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # 图例
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # ---------- 2.6 添加说明文本框（使用默认字体而不是等宽字体） ----------
    description = """
┌─────────────────────────────────────────────────────────────────┐
│  集成响度 (Integrated Loudness, LUFS)                           │
│  ─────────────────────────────────────────────────────────────  │
│  标准: ITU-R BS.1770 | 基于人耳听感的整体感知响度               │
├─────────────────────────────────────────────────────────────────┤
│  参考阈值:                                                       │
│  ● -14 ~ -16 LUFS : 主流音乐推荐，平衡且有存在感                │
│  ● -18 ~ -20 LUFS : 偏小，可能缺乏冲击力                        │
│  ● > -12 LUFS     : 过度压缩，可能失真                          │
│  ● < -22 LUFS     : 能量不足（AI翻唱常见问题）                  │
├─────────────────────────────────────────────────────────────────┤
│  🎯 AI翻唱建议目标: -16 ± 2 LUFS                                 │
│  ✅ 绿色=优秀  🟠 橙色=需关注  🔴 红色=问题                      │
└─────────────────────────────────────────────────────────────────┘
    """

    # 使用常规字体而不是等宽字体
    fig.text(0.02, 0.02, description, fontsize=9,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                       edgecolor='#dee2e6', alpha=0.95))

    # ---------- 2.7 添加统计摘要（使用默认字体） ----------
    avg_lufs = np.mean(results)
    std_lufs = np.std(results)
    best_idx = np.argmin(np.abs(np.array(results) - (-16)))  # 最接近-16的

    summary = f"""
统计摘要:
• 平均值: {avg_lufs:.1f} LUFS
• 标准差: {std_lufs:.2f}
• 最接近理想值: {file_names[best_idx]}
• 样本数: {len(results)}
    """

    fig.text(0.98, 0.02, summary, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f5e9',
                       edgecolor='#a5d6a7', alpha=0.95))

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, left=0.15)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"✅ 图表已保存至: {save_path}")

    if show_plot:
        plt.show()

    plt.close()

    # ========== 3. 返回结果 ==========
    return dict(zip(file_names, results))


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例用法
    wav_files = [
        "path/to/song1.wav",
        "path/to/song2.wav",
        "path/to/song3.wav",
    ]

    # 调用分析函数
    results = analyze_integrated_loudness(wav_files)

    # 打印结果
    print("\n" + "=" * 50)
    print("LUFS 分析结果")
    print("=" * 50)
    for name, lufs in sorted(results.items(), key=lambda x: abs(x[1] - (-16))):
        deviation = lufs - (-16)
        print(f"{name}: {lufs:.1f} LUFS (偏差: {deviation:+.1f})")
