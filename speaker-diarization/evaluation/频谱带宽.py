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

+ 频谱带宽（Spectral Bandwidth）
    + 含义：能量分布宽度，公式：`SB = √[Σ((f - SC)² × M(f)) / Σ M(f)]`；反映声音丰满度。
    + 1500–3000 Hz：正常歌声
    + 过窄：损失信息
    + 过宽：噪声或伪影
"""


# pip install numpy librosa matplotlib soundfile

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
import warnings


def evaluate_spectral_bandwidth(wav_paths: List[str]) -> None:
    """
    评估多个WAV文件的频谱带宽并进行可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数递增排序）
    """

    # 延迟导入，保持函数独立性
    import librosa

    warnings.filterwarnings('ignore')

    # ==================== 内部函数定义 ====================

    def process_single_file(args: Tuple[int, str]) -> Tuple[int, str, Optional[float], Optional[str]]:
        """处理单个音频文件，计算频谱带宽"""
        idx, wav_path = args
        try:
            # 加载音频文件
            y, sr = librosa.load(wav_path, sr=None)

            # 计算频谱带宽（每帧），然后取平均值
            spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            mean_bw = float(np.mean(spectral_bw))

            filename = Path(wav_path).stem
            return (idx, filename, mean_bw, None)
        except Exception as e:
            filename = Path(wav_path).stem
            return (idx, filename, None, str(e))

    def configure_plot_style():
        """配置图表样式：常规字体，中文支持"""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [
            'SimHei', 'Microsoft YaHei', 'PingFang SC',
            'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans'
        ]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10

    # ==================== 并发处理文件 ====================

    print(f"🎵 正在并发处理 {len(wav_paths)} 个音频文件...")

    results: List[Tuple[int, str, Optional[float], Optional[str]]] = []

    with ThreadPoolExecutor() as executor:
        task_args = [(i, path) for i, path in enumerate(wav_paths)]
        futures = [executor.submit(process_single_file, args) for args in task_args]

        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"   进度: {completed}/{len(wav_paths)}")

    # 按原始索引排序（保持轮数递增顺序）
    results.sort(key=lambda x: x[0])

    # 分离有效结果和错误
    valid_results = [(idx, name, val) for idx, name, val, err in results if err is None]
    error_results = [(name, err) for idx, name, val, err in results if err is not None]

    if error_results:
        print(f"\n⚠️  {len(error_results)} 个文件处理失败：")
        for name, err in error_results[:5]:
            print(f"   - {name}: {err}")
        if len(error_results) > 5:
            print(f"   ... 以及其他 {len(error_results) - 5} 个文件")

    if not valid_results:
        print("❌ 没有有效结果可以展示")
        return

    print(f"✅ 成功处理 {len(valid_results)} 个文件，正在生成图表...")

    # 提取数据
    indices = [v[0] for v in valid_results]
    filenames = [v[1] for v in valid_results]
    bandwidths = [v[2] for v in valid_results]

    # ==================== 绑制图表 ====================

    configure_plot_style()

    n_files = len(filenames)

    # 根据文件数量动态调整图表尺寸
    fig_width = min(22, max(14, n_files * 0.18))
    fig_height = 10

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(n_files)

    # ---------- 绘制理想范围区域 ----------
    ax.axhspan(1500, 3000, alpha=0.15, color='#28A745', zorder=1, label='理想范围 (1500-3000 Hz)')
    ax.axhline(1500, color='#28A745', linestyle='--', linewidth=2, alpha=0.8, zorder=2)
    ax.axhline(3000, color='#28A745', linestyle='--', linewidth=2, alpha=0.8, zorder=2)

    # ---------- 绘制数据折线图 ----------
    ax.plot(x, bandwidths,
            linestyle='-', linewidth=1.8,
            marker='o', markersize=5,
            color='#2E86AB',
            markerfacecolor='#2E86AB',
            markeredgecolor='white',
            markeredgewidth=0.8,
            label='频谱带宽',
            zorder=5)

    # 标记超出理想范围的点
    for i, bw in enumerate(bandwidths):
        if bw < 1500:
            ax.scatter(i, bw, color='#DC3545', s=60, zorder=6, marker='v')
        elif bw > 3000:
            ax.scatter(i, bw, color='#FFC107', s=60, zorder=6, marker='^')

    # ---------- 阈值标注 ----------
    ax.annotate('下限 1500 Hz\n(低于此值: 信息损失)',
                xy=(n_files * 0.02, 1500),
                xytext=(n_files * 0.02, 1350),
                fontsize=9, color='#28A745',
                arrowprops=dict(arrowstyle='->', color='#28A745', lw=1),
                va='top')

    ax.annotate('上限 3000 Hz\n(高于此值: 噪声/伪影)',
                xy=(n_files * 0.02, 3000),
                xytext=(n_files * 0.02, 3150),
                fontsize=9, color='#28A745',
                arrowprops=dict(arrowstyle='->', color='#28A745', lw=1),
                va='bottom')

    # ---------- 坐标轴设置 ----------
    ax.set_xlabel('音频文件（按模型训练轮数递增 →）', fontsize=12, fontweight='bold')
    ax.set_ylabel('频谱带宽 (Hz)', fontsize=12, fontweight='bold')
    ax.set_title('AI翻唱质量评估 — 频谱带宽 (Spectral Bandwidth) 趋势分析',
                 fontsize=15, fontweight='bold', pad=20)

    # X轴标签智能处理
    if n_files <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=55, ha='right', fontsize=8)
    else:
        # 大量文件时，均匀选择显示的标签
        step = max(1, n_files // 25)
        visible_indices = list(range(0, n_files, step))
        if (n_files - 1) not in visible_indices:
            visible_indices.append(n_files - 1)
        ax.set_xticks(visible_indices)
        ax.set_xticklabels([filenames[i] for i in visible_indices], rotation=55, ha='right', fontsize=8)

    ax.set_xlim(-0.5, n_files - 0.5)

    # Y轴范围：确保差异可见
    v_min, v_max = min(bandwidths), max(bandwidths)
    v_range = v_max - v_min

    # 如果数据范围太小，适当扩展以显示差异
    if v_range < 200:
        padding = 150
    else:
        padding = v_range * 0.2

    y_lower = min(v_min - padding, 1300)
    y_upper = max(v_max + padding, 3200)
    ax.set_ylim(y_lower, y_upper)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    # ---------- 信息面板（透明背景） ----------
    # 计算统计信息
    mean_bw = np.mean(bandwidths)
    std_bw = np.std(bandwidths)
    in_range_count = sum(1 for bw in bandwidths if 1500 <= bw <= 3000)
    in_range_pct = in_range_count / n_files * 100

    info_text = (
        "【指标说明】\n"
        "─────────────────────\n"
        "频谱带宽 (Spectral Bandwidth)\n\n"
        "◈ 定义\n"
        "   声音能量在频率轴上的分布宽度\n\n"
        "◈ 计算公式\n"
        "   SB = √[Σ((f−SC)²×M(f)) / ΣM(f)]\n"
        "   其中 SC 为频谱质心\n\n"
        "◈ 物理意义\n"
        "   反映声音的丰满度与音色饱满程度\n\n"
        "【阈值标准】\n"
        "─────────────────────\n"
        "✓ 1500−3000 Hz  正常歌声\n"
        "▼ < 1500 Hz     声音单薄/信息损失\n"
        "▲ > 3000 Hz     可能存在噪声或伪影\n\n"
        "【当前数据统计】\n"
        "─────────────────────\n"
        f"  文件总数：{n_files}\n"
        f"  最小值：  {v_min:.1f} Hz\n"
        f"  最大值：  {v_max:.1f} Hz\n"
        f"  平均值：  {mean_bw:.1f} Hz\n"
        f"  标准差：  {std_bw:.1f} Hz\n"
        f"  达标率：  {in_range_pct:.1f}% ({in_range_count}/{n_files})"
    )

    # 文字框 - 透明背景
    ax.text(
        1.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily='sans-serif',
        verticalalignment='top',
        linespacing=1.4,
        bbox=dict(
            boxstyle='round,pad=0.7',
            facecolor='none',  # 透明背景
            edgecolor='#AAAAAA',
            linewidth=1.2
        )
    )

    # 图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # 调整布局，为右侧信息面板留出空间
    plt.tight_layout()
    plt.subplots_adjust(right=0.74)

    print("📊 图表生成完成！")
    plt.show()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例：替换为实际的文件路径列表
    example_paths = [
        "/path/to/model_epoch_100.wav",
        "/path/to/model_epoch_200.wav",
        "/path/to/model_epoch_300.wav",
        # ... 更多文件
    ]

    # 调用函数
    # evaluate_spectral_bandwidth(example_paths)

    print("请提供WAV文件路径列表来调用 evaluate_spectral_bandwidth() 函数")