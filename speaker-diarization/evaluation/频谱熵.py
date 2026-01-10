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

+ 频谱熵（Spectral Entropy）
    + 含义：频谱无序度，公式：Entropy = -Σ (p_i × log₂ p_i)；高值噪声主导，低值谐波有序。
    + 0.4–0.6：平衡有序
    + 0.6–0.8：轻微复杂度
    + 大于0.8 或 小于0.4：噪声重或过度平滑
"""

# pip install numpy librosa matplotlib scipy

"""
AI翻唱质量评估 - 频谱熵分析工具

依赖安装:
pip install numpy librosa matplotlib scipy

使用示例:
    wav_files = ["model_100.wav", "model_200.wav", "model_300.wav", ...]
    analyze_spectral_entropy(wav_files)
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import warnings
from typing import List, Tuple, Optional


def analyze_spectral_entropy(wav_paths: List[str]) -> dict:
    """
    分析多个WAV文件的频谱熵并可视化对比

    频谱熵（Spectral Entropy）衡量音频频谱的无序程度：
    - 高值表示噪声主导，频谱杂乱
    - 低值表示谐波有序，频谱清晰
    - 0.4-0.6 为理想范围，表示平衡有序

    Parameters
    ----------
    wav_paths : List[str]
        WAV文件路径的字符串数组，按模型轮数递增排序

    Returns
    -------
    dict
        包含各文件名及其对应频谱熵值的字典
    """

    warnings.filterwarnings('ignore')

    # ==================== 内部计算函数 ====================
    def _compute_spectral_entropy(wav_path: str) -> Tuple[str, Optional[float], Optional[str]]:
        """
        计算单个WAV文件的归一化频谱熵

        计算公式: H = -Σ(pᵢ × log₂(pᵢ)) / log₂(N)
        其中 pᵢ 为第i个频率分量的能量占比，N为频率分量数量
        """
        filename = os.path.basename(wav_path)
        try:
            # 加载音频文件
            y, sr = librosa.load(wav_path, sr=None)

            # 短时傅里叶变换获取频谱
            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

            # 计算每帧的归一化频谱熵
            frame_entropies = []
            n_bins = S.shape[0]
            max_entropy = np.log2(n_bins)  # 最大可能熵值

            for frame in S.T:
                total_energy = np.sum(frame)
                if total_energy > 1e-10:  # 跳过静音帧
                    # 计算概率分布
                    p = frame / total_energy
                    # 计算熵 (避免log(0))
                    p_nonzero = p[p > 1e-10]
                    entropy_val = -np.sum(p_nonzero * np.log2(p_nonzero))
                    # 归一化到 [0, 1]
                    normalized_entropy = entropy_val / max_entropy if max_entropy > 0 else 0
                    frame_entropies.append(normalized_entropy)

            if not frame_entropies:
                return (filename, None, "音频为静音或无有效帧")

            # 返回整体平均频谱熵
            mean_entropy = float(np.mean(frame_entropies))
            return (filename, mean_entropy, None)

        except Exception as e:
            return (filename, None, str(e))

    # ==================== 并发处理 ====================
    print(f"开始处理 {len(wav_paths)} 个文件...")

    results_map = {}
    max_workers = min(os.cpu_count() or 4, 8, len(wav_paths))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_compute_spectral_entropy, path): path
            for path in wav_paths
        }

        completed = 0
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            results_map[path] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"  进度: {completed}/{len(wav_paths)}")

    # ==================== 整理结果（保持原始顺序）====================
    filenames = []
    entropies = []
    errors = []

    for path in wav_paths:
        name, value, error = results_map[path]
        filenames.append(name)
        entropies.append(value if value is not None else np.nan)
        if error:
            errors.append(f"⚠ {name}: {error}")

    # 打印错误信息
    for err in errors:
        print(err)

    # 检查有效数据
    valid_data = [e for e in entropies if not np.isnan(e)]
    if not valid_data:
        raise ValueError("没有成功处理任何文件，请检查文件路径和格式")

    print(f"成功处理 {len(valid_data)}/{len(wav_paths)} 个文件")

    # ==================== 可视化配置 ====================
    # 设置中文字体（按优先级尝试）
    plt.rcParams.update({
        'font.sans-serif': [
            'Microsoft YaHei', 'SimHei', 'PingFang SC',
            'Hiragino Sans GB', 'WenQuanYi Micro Hei',
            'Noto Sans CJK SC', 'DejaVu Sans', 'Arial'
        ],
        'font.family': 'sans-serif',
        'axes.unicode_minus': False,
        'figure.dpi': 100
    })

    n_files = len(filenames)

    # 动态计算图表尺寸
    fig_width = max(16, min(n_files * 0.4, 45))
    fig_height = 10

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(n_files)

    # ==================== 根据阈值着色 ====================
    def _get_color(val):
        """根据频谱熵值返回对应颜色"""
        if np.isnan(val):
            return '#95a5a6'  # 灰色 - 无效数据
        elif 0.4 <= val <= 0.6:
            return '#27ae60'  # 绿色 - 理想范围
        elif 0.6 < val <= 0.8 or 0.3 <= val < 0.4:
            return '#f39c12'  # 橙色 - 轻微偏离
        else:
            return '#e74c3c'  # 红色 - 需要关注

    colors = [_get_color(e) for e in entropies]

    # ==================== 绘制图表 ====================
    # 趋势线
    ax.plot(x, entropies, color='#3498db', linewidth=1.5,
            alpha=0.6, zorder=2, label='趋势线')

    # 数据点（带颜色标识）
    scatter = ax.scatter(x, entropies, c=colors, s=70, zorder=4,
                         edgecolors='white', linewidths=1)

    # 阈值区域和线
    ax.axhspan(0.4, 0.6, alpha=0.1, color='#27ae60', zorder=1)
    ax.axhline(0.4, color='#27ae60', linestyle='--', linewidth=1.8,
               alpha=0.8, zorder=3)
    ax.axhline(0.6, color='#f39c12', linestyle='--', linewidth=1.8,
               alpha=0.8, zorder=3)
    ax.axhline(0.8, color='#e74c3c', linestyle='--', linewidth=1.8,
               alpha=0.8, zorder=3)

    # 在阈值线旁添加标注
    ax.text(n_files - 0.5, 0.4, '0.4', fontsize=9, color='#27ae60',
            va='bottom', ha='left', fontweight='bold')
    ax.text(n_files - 0.5, 0.6, '0.6', fontsize=9, color='#f39c12',
            va='bottom', ha='left', fontweight='bold')
    ax.text(n_files - 0.5, 0.8, '0.8', fontsize=9, color='#e74c3c',
            va='bottom', ha='left', fontweight='bold')

    # ==================== Y轴范围优化（突出差异）====================
    data_min = min(valid_data)
    data_max = max(valid_data)
    data_range = data_max - data_min

    # 自适应边距，数据范围小时增大边距以显示差异
    if data_range < 0.05:
        margin = 0.08
    elif data_range < 0.1:
        margin = 0.06
    else:
        margin = data_range * 0.12

    # 确保包含重要阈值线
    y_min = max(0, min(data_min - margin, 0.35))
    y_max = min(1.0, max(data_max + margin, 0.85))

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, n_files - 0.5)

    # ==================== X轴标签处理 ====================
    if n_files > 30:
        # 文件多时，间隔显示标签
        step = max(1, n_files // 25)
        tick_positions = list(range(0, n_files, step))
        # 确保最后一个文件显示
        if (n_files - 1) not in tick_positions:
            tick_positions.append(n_files - 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [filenames[i] for i in tick_positions],
            rotation=55, ha='right', fontsize=8
        )
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=55, ha='right', fontsize=9)

    # ==================== 标签和标题 ====================
    ax.set_xlabel('文件（按模型训练轮数递增 →）', fontsize=12, labelpad=10)
    ax.set_ylabel('频谱熵值', fontsize=12)
    ax.set_title('AI翻唱质量评估 — 频谱熵分析 (Spectral Entropy)',
                 fontsize=15, fontweight='bold', pad=15)

    # ==================== 说明文字框（透明背景）====================
    description_text = (
        "【频谱熵 Spectral Entropy】\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "衡量频谱的无序程度\n"
        "公式: H = -Σ(pᵢ × log₂pᵢ)\n\n"
        "• 高值 → 噪声主导，频谱杂乱\n"
        "• 低值 → 谐波有序，频谱清晰\n\n"
        "【质量阈值判断】\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "● 0.4 - 0.6  平衡有序（最佳）\n"
        "● 0.6 - 0.8  轻微复杂\n"
        "● > 0.8      噪声过重\n"
        "● < 0.4      过度平滑"
    )

    ax.text(0.02, 0.97, description_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', linespacing=1.3,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='none',  # 透明背景
                edgecolor='#7f8c8d',
                linewidth=1
            ))

    # ==================== 统计信息框 ====================
    stats_text = (
        f"【统计信息】\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"文件总数:  {n_files}\n"
        f"有效数据:  {len(valid_data)}\n"
        f"平均值:    {np.nanmean(entropies):.4f}\n"
        f"最小值:    {data_min:.4f}\n"
        f"最大值:    {data_max:.4f}\n"
        f"标准差:    {np.nanstd(entropies):.4f}\n"
        f"数据范围:  {data_range:.4f}"
    )

    ax.text(0.98, 0.97, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            linespacing=1.3,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='none',  # 透明背景
                edgecolor='#7f8c8d',
                linewidth=1
            ))

    # ==================== 图例 ====================
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='white',
              label='理想范围 (0.4-0.6)'),
        Patch(facecolor='#f39c12', edgecolor='white',
              label='轻微偏离 (0.3-0.4 / 0.6-0.8)'),
        Patch(facecolor='#e74c3c', edgecolor='white',
              label='需要关注 (<0.3 / >0.8)'),
        Line2D([0], [0], color='#3498db', linewidth=2,
               label='变化趋势'),
        Patch(facecolor='#27ae60', alpha=0.2, edgecolor='none',
              label='理想区间'),
    ]

    ax.legend(handles=legend_elements,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.12),
              ncol=5, fontsize=9,
              frameon=True, fancybox=True,
              shadow=False)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # ==================== 布局调整与显示 ====================
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.92)

    print("\n正在显示图表...")
    plt.show()

    # 返回结果字典
    return dict(zip(filenames, entropies))


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例用法
    import glob

    # 方式1: 指定文件列表
    # wav_files = [
    #     "outputs/model_epoch100.wav",
    #     "outputs/model_epoch200.wav",
    #     "outputs/model_epoch300.wav",
    #     # ...更多文件
    # ]

    # 方式2: 使用glob匹配
    # wav_files = sorted(glob.glob("outputs/*.wav"))

    # 调用分析函数
    # results = analyze_spectral_entropy(wav_files)
    # print("\n各文件频谱熵值:", results)

    print("请提供WAV文件路径列表来运行分析")