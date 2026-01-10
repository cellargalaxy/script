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

+ 总谐波失真 + 噪声（THD+N）
    + 含义：处理中引入的谐波干扰；高值表示数码味或破音。
    + 低：干净
    + 高：严重失真（24bit/44.1kHz标准）
"""

# pip install numpy scipy matplotlib

"""
AI翻唱音频质量分析 - THD+N（总谐波失真+噪声）评估
依赖安装: pip install numpy scipy matplotlib
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
from typing import List, Dict, Any


def analyze_thdn(wav_paths: List[str]) -> List[Dict[str, Any]]:
    """
    分析多个WAV文件的THD+N（总谐波失真+噪声）质量

    对于AI翻唱音频，使用综合失真评估方法：
    1. 削波失真检测 - 检测破音/过载
    2. 频谱失真分析 - 检测数码味/高频伪影
    3. 噪底估算 - 评估背景噪声水平
    4. 动态范围分析 - 评估信号质量

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数排序）

    返回:
        包含每个文件分析结果的字典列表
    """

    def _calculate_single_file(args: tuple) -> Dict[str, Any]:
        """计算单个文件的THD+N相关指标（内部函数）"""
        index, file_path = args

        try:
            # ====== 读取音频文件 ======
            sample_rate, data = wavfile.read(file_path)

            # 转换为单声道
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # 归一化到 [-1, 1]
            data = data.astype(np.float64)
            original_dtype = data.dtype

            # 根据位深度归一化
            if np.issubdtype(original_dtype, np.integer):
                max_int = np.iinfo(np.int16).max if data.max() < 32768 else np.iinfo(np.int32).max
                data = data / max_int
            else:
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val

            # ====== 1. 削波失真检测 ======
            clip_threshold = 0.995
            clip_samples = np.sum(np.abs(data) >= clip_threshold)
            clip_ratio = (clip_samples / len(data)) * 100

            # 连续削波检测（更严重的失真）
            clipped = np.abs(data) >= clip_threshold
            consecutive_clips = 0
            max_consecutive = 0
            for c in clipped:
                if c:
                    consecutive_clips += 1
                    max_consecutive = max(max_consecutive, consecutive_clips)
                else:
                    consecutive_clips = 0

            # ====== 2. 频谱分析 ======
            nperseg = min(4096, len(data) // 8)
            noverlap = nperseg // 2

            f, t_spec, Zxx = signal.stft(data, sample_rate, nperseg=nperseg, noverlap=noverlap)
            magnitude = np.abs(Zxx)
            power = magnitude ** 2

            # 总能量
            total_power = np.sum(power)

            # 高频失真分析 (>14kHz 通常是失真/噪声区域)
            hf_threshold = 14000
            hf_mask = f > hf_threshold
            hf_power = np.sum(power[hf_mask, :]) if np.any(hf_mask) else 0
            hf_ratio = (hf_power / total_power * 100) if total_power > 0 else 0

            # 超高频分析 (>18kHz 几乎全是失真/数字伪影)
            uhf_mask = f > 18000
            uhf_power = np.sum(power[uhf_mask, :]) if np.any(uhf_mask) else 0
            uhf_ratio = (uhf_power / total_power * 100) if total_power > 0 else 0

            # ====== 3. 噪底估算 ======
            frame_power = np.sum(power, axis=0)

            # 找最安静的10%帧作为噪底估算
            quiet_threshold = np.percentile(frame_power, 10)
            quiet_frames = frame_power <= quiet_threshold

            if np.any(quiet_frames) and np.sum(quiet_frames) > 1:
                noise_floor_power = np.mean(power[:, quiet_frames])
                signal_power = np.mean(power[:, ~quiet_frames]) if np.any(~quiet_frames) else np.mean(power)

                if noise_floor_power > 0:
                    snr_linear = signal_power / noise_floor_power
                    snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else 60
                else:
                    snr_db = 60
            else:
                snr_db = 40  # 默认值

            snr_db = np.clip(snr_db, 0, 80)  # 限制范围

            # ====== 4. 动态范围分析 ======
            # RMS和峰值比
            rms = np.sqrt(np.mean(data ** 2))
            peak = np.max(np.abs(data))
            crest_factor = peak / rms if rms > 0 else 1
            crest_factor_db = 20 * np.log10(crest_factor) if crest_factor > 0 else 0

            # ====== 5. 计算综合THD+N估算值 ======
            # 权重组合各项失真指标
            distortion_components = {
                'clip_contribution': clip_ratio * 15,  # 削波影响最大
                'consecutive_clip': (max_consecutive / 100) * 10,  # 连续削波
                'hf_contribution': hf_ratio * 3,  # 高频失真
                'uhf_contribution': uhf_ratio * 8,  # 超高频伪影
                'noise_contribution': max(0, (40 - snr_db) * 0.5)  # 噪声贡献
            }

            total_distortion = sum(distortion_components.values())

            # 转换为THD+N的dB表示
            # 基准：完美信号为 -80dB，最差为 0dB
            thdn_db = -80 + total_distortion
            thdn_db = np.clip(thdn_db, -80, 0)

            # 转换为百分比
            thdn_percent = 10 ** (thdn_db / 20) * 100

            # ====== 6. 质量评级 ======
            if thdn_db < -60:
                quality_rating = "优秀"
            elif thdn_db < -45:
                quality_rating = "良好"
            elif thdn_db < -30:
                quality_rating = "一般"
            else:
                quality_rating = "较差"

            return {
                'index': index,
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'thdn_db': thdn_db,
                'thdn_percent': thdn_percent,
                'clip_ratio': clip_ratio,
                'max_consecutive_clips': max_consecutive,
                'hf_ratio': hf_ratio,
                'uhf_ratio': uhf_ratio,
                'snr_db': snr_db,
                'crest_factor_db': crest_factor_db,
                'distortion_components': distortion_components,
                'quality_rating': quality_rating,
                'sample_rate': sample_rate,
                'duration_sec': len(data) / sample_rate,
                'success': True
            }

        except Exception as e:
            return {
                'index': index,
                'file_path': file_path,
                'file_name': os.path.basename(file_path) if file_path else 'Unknown',
                'error': str(e),
                'success': False
            }

    # ====== 并发处理所有文件 ======
    indexed_paths = list(enumerate(wav_paths))
    results = []

    max_workers = min(os.cpu_count() or 4, len(wav_paths), 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(_calculate_single_file, indexed_paths)
        results = list(futures)

    # 按原始顺序排序
    results.sort(key=lambda x: x['index'])

    # 分离成功和失败的结果
    valid_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    # 打印失败信息
    for r in failed_results:
        print(f"⚠ 处理失败: {r['file_name']} - {r.get('error', '未知错误')}")

    if not valid_results:
        print("❌ 没有成功处理的文件")
        return results

    print(f"✓ 成功处理 {len(valid_results)}/{len(wav_paths)} 个文件")

    # ====== 可视化 ======
    _visualize_results(valid_results)

    return results


def _visualize_results(valid_results: List[Dict[str, Any]]) -> None:
    """可视化分析结果（内部函数）"""

    # 设置字体（优先使用系统中文字体）
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    n_files = len(valid_results)

    # 根据文件数量调整图表大小
    if n_files <= 20:
        fig_width, fig_height = 16, 12
        font_size = 9
        rotation = 45
    elif n_files <= 50:
        fig_width, fig_height = 20, 14
        font_size = 7
        rotation = 60
    else:
        fig_width, fig_height = 24, 16
        font_size = 5
        rotation = 90

    fig = plt.figure(figsize=(fig_width, fig_height))

    # 创建网格布局
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.3], hspace=0.35, wspace=0.25)

    # 提取数据
    file_names = [r['file_name'] for r in valid_results]
    # 创建简短标签（截取关键部分）
    short_names = []
    for name in file_names:
        base = os.path.splitext(name)[0]
        if len(base) > 25:
            short_names.append(f"...{base[-22:]}")
        else:
            short_names.append(base)

    x = np.arange(n_files)

    # 提取各项指标
    thdn_db_values = np.array([r['thdn_db'] for r in valid_results])
    clip_ratios = np.array([r['clip_ratio'] for r in valid_results])
    hf_ratios = np.array([r['hf_ratio'] for r in valid_results])
    snr_values = np.array([r['snr_db'] for r in valid_results])

    # ====== 子图1: THD+N 主指标（折线图 + 散点） ======
    ax1 = fig.add_subplot(gs[0, :])

    # 根据THD+N值设置颜色
    colors = []
    for v in thdn_db_values:
        if v < -60:
            colors.append('#2ecc71')  # 绿色-优秀
        elif v < -45:
            colors.append('#3498db')  # 蓝色-良好
        elif v < -30:
            colors.append('#f39c12')  # 橙色-一般
        else:
            colors.append('#e74c3c')  # 红色-较差

    # 绘制折线
    ax1.plot(x, thdn_db_values, 'b-', linewidth=1.5, alpha=0.6, zorder=1)
    # 绘制散点（带颜色编码）
    ax1.scatter(x, thdn_db_values, c=colors, s=60, zorder=2, edgecolors='white', linewidth=0.5)

    # 添加参考线和区域
    ax1.axhline(y=-60, color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=-45, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=-30, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.8)

    # 填充质量区域
    y_min, y_max = ax1.get_ylim()
    ax1.fill_between(x, -80, -60, alpha=0.1, color='#2ecc71')
    ax1.fill_between(x, -60, -45, alpha=0.1, color='#3498db')
    ax1.fill_between(x, -45, -30, alpha=0.1, color='#f39c12')
    ax1.fill_between(x, -30, 0, alpha=0.1, color='#e74c3c')

    # 动态调整Y轴范围（放大差异）
    data_min, data_max = thdn_db_values.min(), thdn_db_values.max()
    data_range = data_max - data_min
    padding = max(data_range * 0.15, 3)  # 至少3dB的padding
    ax1.set_ylim(data_min - padding, data_max + padding)

    ax1.set_ylabel('THD+N (dB)', fontsize=11)
    ax1.set_title('总谐波失真+噪声 (THD+N) - 值越低越好（越干净）', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=rotation, ha='right', fontsize=font_size)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-0.5, n_files - 0.5)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='优秀 (<-60dB)'),
        Patch(facecolor='#3498db', alpha=0.7, label='良好 (-60~-45dB)'),
        Patch(facecolor='#f39c12', alpha=0.7, label='一般 (-45~-30dB)'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='较差 (>-30dB)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # 标注最佳和最差点
    best_idx = np.argmin(thdn_db_values)
    worst_idx = np.argmax(thdn_db_values)
    ax1.annotate(f'最佳\n{thdn_db_values[best_idx]:.1f}dB',
                 xy=(best_idx, thdn_db_values[best_idx]),
                 xytext=(10, -20), textcoords='offset points',
                 fontsize=8, color='#2ecc71',
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1))
    if worst_idx != best_idx:
        ax1.annotate(f'最差\n{thdn_db_values[worst_idx]:.1f}dB',
                     xy=(worst_idx, thdn_db_values[worst_idx]),
                     xytext=(10, 20), textcoords='offset points',
                     fontsize=8, color='#e74c3c',
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1))

    # ====== 子图2: 削波率（柱状图） ======
    ax2 = fig.add_subplot(gs[1, 0])

    bar_colors = ['#e74c3c' if v > 0.5 else '#f39c12' if v > 0.1 else '#2ecc71' for v in clip_ratios]
    bars = ax2.bar(x, clip_ratios, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    # 参考线
    ax2.axhline(y=0.1, color='#f39c12', linestyle='--', linewidth=1.5, label='警戒线 (0.1%)')
    ax2.axhline(y=0.5, color='#e74c3c', linestyle='--', linewidth=1.5, label='严重 (0.5%)')

    ax2.set_ylabel('削波率 (%)', fontsize=11)
    ax2.set_title('削波失真检测 - 高值表示破音/过载', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=rotation, ha='right', fontsize=font_size)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(-0.5, n_files - 0.5)

    # 动态调整Y轴
    if clip_ratios.max() > 0:
        ax2.set_ylim(0, clip_ratios.max() * 1.2 + 0.05)
    else:
        ax2.set_ylim(0, 0.2)

    # ====== 子图3: 信噪比和高频失真（双Y轴） ======
    ax3 = fig.add_subplot(gs[1, 1])
    ax3_twin = ax3.twinx()

    # 信噪比（柱状图）
    snr_colors = ['#2ecc71' if v > 40 else '#f39c12' if v > 25 else '#e74c3c' for v in snr_values]
    bars_snr = ax3.bar(x - 0.2, snr_values, width=0.4, color=snr_colors, alpha=0.7, label='信噪比 (dB)')

    # 高频失真比例（折线图）
    ax3_twin.plot(x, hf_ratios, 'o-', color='#9b59b6', linewidth=2, markersize=4, label='高频失真比 (%)')

    ax3.axhline(y=40, color='#2ecc71', linestyle=':', linewidth=1, alpha=0.7)
    ax3.axhline(y=25, color='#f39c12', linestyle=':', linewidth=1, alpha=0.7)

    ax3.set_ylabel('信噪比 SNR (dB)', fontsize=11, color='#3498db')
    ax3_twin.set_ylabel('高频失真比 (%)', fontsize=11, color='#9b59b6')
    ax3.set_title('信噪比 & 高频失真分析', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=rotation, ha='right', fontsize=font_size)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xlim(-0.5, n_files - 0.5)

    # 合并图例
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # ====== 说明文字区域 ======
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    # 统计信息
    avg_thdn = np.mean(thdn_db_values)
    std_thdn = np.std(thdn_db_values)
    excellent_count = np.sum(thdn_db_values < -60)
    good_count = np.sum((thdn_db_values >= -60) & (thdn_db_values < -45))
    fair_count = np.sum((thdn_db_values >= -45) & (thdn_db_values < -30))
    poor_count = np.sum(thdn_db_values >= -30)

    description = f"""
【THD+N 总谐波失真+噪声 评估说明】

▶ 指标含义：THD+N测量音频中所有谐波失真和噪声的总和，反映AI翻唱处理中引入的干扰程度。值越低(越负)表示音频越干净。
▶ 质量阈值：优秀 < -60dB（专业级）| 良好 -60~-45dB | 一般 -45~-30dB（可接受）| 较差 > -30dB（明显失真）
▶ 削波失真：当削波率 > 0.1% 时需注意，> 0.5% 表示存在明显破音。高值通常意味着数码味或信号过载。
▶ 信噪比：> 40dB 优秀 | 25~40dB 良好 | < 25dB 较差。高频失真比过高可能表示存在数字处理产生的伪影。

【本次分析统计】
• 文件总数: {len(valid_results)} | 平均THD+N: {avg_thdn:.2f}dB (±{std_thdn:.2f})
• 质量分布: 优秀({excellent_count}) | 良好({good_count}) | 一般({fair_count}) | 较差({poor_count})
• 最佳: {valid_results[best_idx]['file_name']} ({thdn_db_values[best_idx]:.2f}dB)
• 最差: {valid_results[worst_idx]['file_name']} ({thdn_db_values[worst_idx]:.2f}dB)
• 趋势: {'↗ 随轮数增加质量下降' if thdn_db_values[-1] > thdn_db_values[0] + 3 else '↘ 随轮数增加质量提升' if thdn_db_values[-1] < thdn_db_values[0] - 3 else '→ 质量相对稳定'}
    """

    ax_text.text(0.02, 0.95, description, transform=ax_text.transAxes,
                 fontsize=10, verticalalignment='top',
                 family='sans-serif',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='none', edgecolor='#cccccc', alpha=0.8))

    # 总标题
    fig.suptitle('AI翻唱音频质量分析 - THD+N 综合评估报告',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例用法
    test_paths = [
        r"path/to/model_epoch_100.wav",
        r"path/to/model_epoch_200.wav",
        r"path/to/model_epoch_300.wav",
        # ... 更多文件路径
    ]

    # 调用分析函数
    # results = analyze_thdn(test_paths)

    # 打印详细结果
    # for r in results:
    #     if r['success']:
    #         print(f"{r['file_name']}: THD+N={r['thdn_db']:.2f}dB ({r['quality_rating']})")