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

+ 振幅微扰（Shimmer）
    + 含义：连续周期振幅变化，公式：`Shimmer(%) = [Σ|A(i) - A(i+1)| / ((N-1) × 平均振幅)] × 100%`；反映能量稳定性。
    + 小于3%：稳定
    + 3–5%：正常
    + 5–10%：不稳
    + 大于10%：明显波动
"""

# pip install praat-parselmouth numpy matplotlib

"""
振幅微扰（Shimmer）质量分析工具

依赖安装：
pip install praat-parselmouth numpy matplotlib

注意：
- praat-parselmouth 是 Praat 语音分析软件的 Python 接口
- 支持 Windows/Linux/macOS
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict


# 在函数外部定义计算函数，确保可被并发调用
def _calculate_shimmer_single(wav_path: str) -> Tuple[str, Optional[float], Optional[str]]:
    """
    计算单个WAV文件的Shimmer值

    返回: (文件名, Shimmer百分比值, 错误信息)
    """
    try:
        import parselmouth
        from parselmouth.praat import call

        filename = os.path.basename(wav_path)

        # 加载音频
        sound = parselmouth.Sound(wav_path)

        # 创建 PointProcess 对象（用于检测基频周期）
        # 参数: 最低基频75Hz, 最高基频600Hz
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

        # 计算 Shimmer (local)
        # 参数说明:
        # - 时间范围: 0, 0 表示整个文件
        # - 最短周期: 0.0001秒
        # - 最长周期: 0.02秒
        # - 最大周期因子: 1.3
        # - 最大振幅因子: 1.6
        shimmer = call([sound, point_process], "Get shimmer (local)",
                       0, 0, 0.0001, 0.02, 1.3, 1.6)

        if shimmer is None or np.isnan(shimmer):
            return (filename, None, "无法计算Shimmer值（可能缺少周期性信号）")

        # 转换为百分比
        shimmer_percent = shimmer * 100

        return (filename, shimmer_percent, None)

    except Exception as e:
        filename = os.path.basename(wav_path)
        return (filename, None, str(e))


def analyze_shimmer_quality(wav_paths: List[str]) -> Dict:
    """
    分析多个WAV文件的振幅微扰（Shimmer）指标，并可视化对比

    ════════════════════════════════════════════════════════════════
    振幅微扰（Shimmer）指标说明：
    ────────────────────────────────────────────────────────────────
    • 定义：连续声周期之间振幅变化的相对量度
    • 公式：Shimmer(%) = [Σ|A(i)-A(i+1)| / ((N-1)×平均振幅)] × 100%
    • 意义：反映声音能量的稳定性，值越低越好

    评判标准：
    • < 3%    ：稳定（优秀）
    • 3% - 5% ：正常（良好）
    • 5% - 10%：不稳（一般）
    • > 10%   ：明显波动（较差）
    ════════════════════════════════════════════════════════════════

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数递增排序）

    返回:
        包含分析结果的字典
    """

    # 检查依赖
    try:
        import parselmouth
    except ImportError:
        raise ImportError(
            "请安装 praat-parselmouth 库:\n"
            "pip install praat-parselmouth"
        )

    if not wav_paths:
        raise ValueError("wav_paths 不能为空")

    # ═══════════════════════════════════════════════════════════════
    # 并发计算所有文件的 Shimmer 值
    # ═══════════════════════════════════════════════════════════════

    num_files = len(wav_paths)
    num_workers = min(os.cpu_count() or 4, num_files, 8)  # 限制最大并发数

    print(f"{'═' * 60}")
    print(f"开始分析 {num_files} 个WAV文件的振幅微扰（Shimmer）...")
    print(f"并发线程数: {num_workers}")
    print(f"{'═' * 60}")

    results: Dict[str, Tuple[str, Optional[float], Optional[str]]] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(_calculate_shimmer_single, path): path
            for path in wav_paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            completed += 1

            try:
                result = future.result()
                results[path] = result

                filename, shimmer_val, error = result
                if shimmer_val is not None:
                    # 根据值确定质量等级
                    if shimmer_val < 3:
                        quality = "稳定 ✓"
                    elif shimmer_val < 5:
                        quality = "正常"
                    elif shimmer_val < 10:
                        quality = "不稳 ⚠"
                    else:
                        quality = "波动 ✗"
                    print(f"[{completed:3d}/{num_files}] {filename}: {shimmer_val:.2f}% ({quality})")
                else:
                    print(f"[{completed:3d}/{num_files}] {filename}: 计算失败 - {error}")

            except Exception as e:
                results[path] = (os.path.basename(path), None, str(e))
                print(f"[{completed:3d}/{num_files}] {os.path.basename(path)}: 异常 - {e}")

    # ═══════════════════════════════════════════════════════════════
    # 按原始顺序整理结果
    # ═══════════════════════════════════════════════════════════════

    ordered_results = [results[path] for path in wav_paths]
    filenames = [r[0] for r in ordered_results]
    shimmer_values = [r[1] for r in ordered_results]
    errors = [r[2] for r in ordered_results]

    valid_values = [v for v in shimmer_values if v is not None]

    print(f"\n{'═' * 60}")
    print(f"分析完成！成功: {len(valid_values)}/{num_files}")
    if valid_values:
        print(f"Shimmer 范围: {min(valid_values):.2f}% - {max(valid_values):.2f}%")
        print(f"Shimmer 均值: {np.mean(valid_values):.2f}%")
        print(f"Shimmer 标准差: {np.std(valid_values):.2f}%")
    print(f"{'═' * 60}\n")

    # ═══════════════════════════════════════════════════════════════
    # 可视化
    # ═══════════════════════════════════════════════════════════════

    # 设置中文字体（尝试多种字体以确保兼容性）
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei',  # Windows
        'SimHei',  # Windows
        'PingFang SC',  # macOS
        'Hiragino Sans GB',  # macOS
        'WenQuanYi Micro Hei',  # Linux
        'Noto Sans CJK SC',  # Linux
        'DejaVu Sans',  # 备选
        'Arial'  # 备选
    ]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

    # 根据文件数量动态调整图表尺寸
    if num_files <= 20:
        fig_width, fig_height = 14, 9
        marker_size = 50
        line_width = 2
        font_size_tick = 9
        rotation = 45
    elif num_files <= 50:
        fig_width, fig_height = 18, 10
        marker_size = 35
        line_width = 1.5
        font_size_tick = 8
        rotation = 60
    else:
        fig_width, fig_height = 24, 12
        marker_size = 20
        line_width = 1.2
        font_size_tick = 7
        rotation = 75

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')

    # 准备绑定数据
    x = np.arange(num_files)
    y = np.array([v if v is not None else np.nan for v in shimmer_values])
    valid_mask = ~np.isnan(y)

    # ─────────────────────────────────────────────────────────────
    # 绘制背景区域（质量分区）
    # ─────────────────────────────────────────────────────────────

    if valid_values:
        y_max = max(max(valid_values) * 1.3, 12)
    else:
        y_max = 15

    # 质量分区背景
    ax.axhspan(0, 3, alpha=0.20, color='#2ecc71', label='稳定区 (<3%) - 优秀')
    ax.axhspan(3, 5, alpha=0.20, color='#f1c40f', label='正常区 (3-5%) - 良好')
    ax.axhspan(5, 10, alpha=0.20, color='#e67e22', label='不稳区 (5-10%) - 一般')
    ax.axhspan(10, y_max, alpha=0.20, color='#e74c3c', label='波动区 (>10%) - 较差')

    # 阈值线
    ax.axhline(y=3, color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=5, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.axhline(y=10, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.8)

    # ─────────────────────────────────────────────────────────────
    # 绘制数据点和折线
    # ─────────────────────────────────────────────────────────────

    # 根据值着色
    colors = []
    for v in shimmer_values:
        if v is None:
            colors.append('#95a5a6')  # 灰色表示无效
        elif v < 3:
            colors.append('#2ecc71')  # 绿色
        elif v < 5:
            colors.append('#f1c40f')  # 黄色
        elif v < 10:
            colors.append('#e67e22')  # 橙色
        else:
            colors.append('#e74c3c')  # 红色

    # 绘制折线（仅连接有效点）
    if np.sum(valid_mask) > 1:
        ax.plot(x[valid_mask], y[valid_mask],
                color='#3498db', linewidth=line_width, alpha=0.7, zorder=3)

    # 绘制散点
    scatter = ax.scatter(x[valid_mask], y[valid_mask],
                         c=[colors[i] for i in range(num_files) if valid_mask[i]],
                         s=marker_size, zorder=5, edgecolors='white', linewidths=0.5)

    # 标记失败的点
    failed_mask = ~valid_mask
    if np.any(failed_mask):
        ax.scatter(x[failed_mask], [0.5] * np.sum(failed_mask),
                   c='#e74c3c', s=marker_size, marker='x',
                   linewidths=2, zorder=5, label='计算失败')

    # ─────────────────────────────────────────────────────────────
    # 坐标轴设置
    # ─────────────────────────────────────────────────────────────

    ax.set_xlim(-0.5, num_files - 0.5)

    # Y轴范围：动态调整以更明显展示差异
    if valid_values:
        val_min = min(valid_values)
        val_max = max(valid_values)
        val_range = val_max - val_min

        if val_range < 1:
            # 差异很小时，放大显示
            padding = 1
        else:
            padding = val_range * 0.15

        y_bottom = max(0, val_min - padding)
        y_top = max(val_max + padding, 6)  # 至少显示到6%以展示正常阈值

        # 确保能看到关键阈值线
        if val_max > 3:
            y_top = max(y_top, val_max + padding)

        ax.set_ylim(y_bottom, y_top)
    else:
        ax.set_ylim(0, 15)

    # X轴标签
    ax.set_xlabel('文件（按模型训练轮数递增 →）', fontsize=12, fontweight='bold')
    ax.set_ylabel('振幅微扰 Shimmer (%)', fontsize=12, fontweight='bold')

    # 标题
    ax.set_title('AI翻唱质量分析 - 振幅微扰（Shimmer）\n值越低表示声音能量越稳定',
                 fontsize=14, fontweight='bold', pad=20)

    # X轴刻度标签
    if num_files <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=rotation, ha='right', fontsize=font_size_tick)
    else:
        # 文件太多时，间隔显示标签
        step = max(1, num_files // 25)
        tick_positions = list(range(0, num_files, step))
        if (num_files - 1) not in tick_positions:
            tick_positions.append(num_files - 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([filenames[i] for i in tick_positions],
                           rotation=rotation, ha='right', fontsize=font_size_tick)

    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)

    # ─────────────────────────────────────────────────────────────
    # 图例
    # ─────────────────────────────────────────────────────────────

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
              fancybox=True, shadow=True)

    # ─────────────────────────────────────────────────────────────
    # 添加说明文本框（透明背景）
    # ─────────────────────────────────────────────────────────────

    description = (
        "【振幅微扰 Shimmer 指标说明】\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "定义：连续声周期振幅变化的相对量度\n"
        "公式：Shimmer = Σ|A(i)-A(i+1)| / ((N-1)×均幅) × 100%\n"
        "意义：反映声音能量稳定性，值越低越好\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "阈值：<3%稳定 | 3-5%正常 | 5-10%不稳 | >10%波动"
    )

    # 文本框样式：背景透明
    text_props = dict(
        boxstyle='round,pad=0.6',
        facecolor='none',  # 透明背景
        edgecolor='#bdc3c7',
        linewidth=1
    )

    ax.text(0.02, 0.97, description,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=text_props,
            family='sans-serif',
            linespacing=1.4)

    # ─────────────────────────────────────────────────────────────
    # 添加统计信息（右下角）
    # ─────────────────────────────────────────────────────────────

    if valid_values:
        stats_text = (
            f"【统计信息】\n"
            f"有效文件: {len(valid_values)}/{num_files}\n"
            f"最小值: {min(valid_values):.2f}%\n"
            f"最大值: {max(valid_values):.2f}%\n"
            f"均值: {np.mean(valid_values):.2f}%\n"
            f"标准差: {np.std(valid_values):.2f}%"
        )

        stats_props = dict(
            boxstyle='round,pad=0.5',
            facecolor='none',
            edgecolor='#bdc3c7',
            linewidth=1
        )

        ax.text(0.98, 0.02, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=stats_props,
                family='sans-serif')

    # ─────────────────────────────────────────────────────────────
    # 标注最优和最差点
    # ─────────────────────────────────────────────────────────────

    if len(valid_values) >= 2:
        valid_indices = [i for i, v in enumerate(shimmer_values) if v is not None]
        valid_vals = [shimmer_values[i] for i in valid_indices]

        best_idx = valid_indices[np.argmin(valid_vals)]
        worst_idx = valid_indices[np.argmax(valid_vals)]

        # 标注最优点
        ax.annotate(f'最优: {shimmer_values[best_idx]:.2f}%\n{filenames[best_idx]}',
                    xy=(best_idx, shimmer_values[best_idx]),
                    xytext=(15, 25), textcoords='offset points',
                    fontsize=8, color='#27ae60',
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#27ae60', alpha=0.9))

        # 标注最差点（仅当与最优不同时）
        if worst_idx != best_idx:
            ax.annotate(f'最差: {shimmer_values[worst_idx]:.2f}%\n{filenames[worst_idx]}',
                        xy=(worst_idx, shimmer_values[worst_idx]),
                        xytext=(15, -35), textcoords='offset points',
                        fontsize=8, color='#c0392b',
                        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='#c0392b', alpha=0.9))

    # 调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()

    # ═══════════════════════════════════════════════════════════════
    # 返回结果字典
    # ═══════════════════════════════════════════════════════════════

    return {
        'filenames': filenames,
        'shimmer_values': shimmer_values,
        'errors': errors,
        'statistics': {
            'valid_count': len(valid_values),
            'total_count': num_files,
            'min': min(valid_values) if valid_values else None,
            'max': max(valid_values) if valid_values else None,
            'mean': np.mean(valid_values) if valid_values else None,
            'std': np.std(valid_values) if valid_values else None
        }
    }


# ═══════════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 示例：分析指定目录下的所有WAV文件
    import glob

    # 方式1：指定文件列表
    wav_files = [
        r"path/to/model_epoch_100.wav",
        r"path/to/model_epoch_200.wav",
        r"path/to/model_epoch_300.wav",
        # ... 更多文件
    ]

    # 方式2：通过glob获取目录下所有WAV文件并排序
    # wav_files = sorted(glob.glob(r"D:\ai_covers\*.wav"))

    # 执行分析
    if wav_files:
        results = analyze_shimmer_quality(wav_files)

        # 可以进一步处理返回的结果
        print("\n最终结果摘要:")
        for fname, shimmer in zip(results['filenames'], results['shimmer_values']):
            if shimmer is not None:
                print(f"  {fname}: {shimmer:.2f}%")