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

+ 信噪比估算（Signal-to-Noise Ratio, SNR）
    + 含义：信号/噪声功率比，公式：SNR(dB) = 10 × log₁₀ (信号能量 / 噪声能量)，通过静音段估算；检测底噪。
    + 大于40 dB：无噪
    + 30–40 dB：轻微噪
    + 20–30 dB：可感知噪
    + 小于20 dB：明显干扰
"""

# pip install numpy scipy matplotlib

"""
AI翻唱音频质量评估 - 信噪比(SNR)分析工具
依赖安装: pip install numpy scipy matplotlib
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple


def analyze_audio_snr(wav_paths: List[str]) -> Dict[str, Optional[float]]:
    """
    分析多个WAV文件的信噪比(SNR)并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（按模型轮数递增排序）

    返回:
        dict: {文件名: SNR值} 的字典
    """

    # ==================== 内部函数定义 ====================

    def _read_wav_as_float(wav_path: str) -> Tuple[int, np.ndarray]:
        """读取WAV文件并转换为归一化浮点数组"""
        sample_rate, audio = wavfile.read(wav_path)

        # 根据数据类型归一化
        if audio.dtype == np.int16:
            audio = audio.astype(np.float64) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float64) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float64) - 128.0) / 128.0
        elif audio.dtype in [np.float32, np.float64]:
            audio = audio.astype(np.float64)
        else:
            audio = audio.astype(np.float64) / np.max(np.abs(audio) + 1e-10)

        # 多声道转单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return sample_rate, audio

    def _calculate_frame_energies(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """分帧计算能量"""
        frame_duration = 0.025  # 25ms帧长
        hop_duration = 0.010  # 10ms步长

        frame_length = int(frame_duration * sample_rate)
        hop_length = int(hop_duration * sample_rate)

        if frame_length <= 0 or hop_length <= 0:
            return np.array([np.mean(audio ** 2)])

        n_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
        frame_energies = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            if end > start:
                frame = audio[start:end]
                frame_energies[i] = np.mean(frame ** 2)

        return frame_energies

    def _estimate_snr(frame_energies: np.ndarray) -> float:
        """通过静音段估算SNR"""
        if len(frame_energies) == 0:
            return 0.0

        sorted_energies = np.sort(frame_energies)

        # 取能量最低的10%帧作为噪声估计
        noise_percentile = 0.10
        noise_frame_count = max(1, int(len(sorted_energies) * noise_percentile))
        noise_energy = np.mean(sorted_energies[:noise_frame_count])

        # 取能量最高的50%帧作为信号估计
        signal_percentile = 0.50
        signal_frame_count = max(1, int(len(sorted_energies) * signal_percentile))
        signal_energy = np.mean(sorted_energies[-signal_frame_count:])

        # 防止除零
        noise_energy = max(noise_energy, 1e-12)
        signal_energy = max(signal_energy, 1e-12)

        # 计算SNR
        snr = 10 * np.log10(signal_energy / noise_energy)

        # 限制合理范围
        return float(np.clip(snr, 0, 80))

    def _calculate_snr_single(wav_path: str) -> Dict:
        """计算单个文件的SNR"""
        result = {
            'path': wav_path,
            'filename': os.path.basename(wav_path),
            'snr': None,
            'error': None
        }

        try:
            sample_rate, audio = _read_wav_as_float(wav_path)

            if len(audio) < sample_rate * 0.1:  # 至少0.1秒
                result['error'] = "音频过短"
                return result

            frame_energies = _calculate_frame_energies(audio, sample_rate)
            snr = _estimate_snr(frame_energies)

            result['snr'] = snr

        except Exception as e:
            result['error'] = str(e)

        return result

    def _get_snr_color(snr: float) -> str:
        """根据SNR值返回对应颜色"""
        if snr >= 40:
            return '#2E7D32'  # 深绿
        elif snr >= 30:
            return '#FF9800'  # 橙色
        elif snr >= 20:
            return '#FFC107'  # 琥珀色
        else:
            return '#D32F2F'  # 红色

    def _get_quality_label(snr: float) -> str:
        """根据SNR值返回质量标签"""
        if snr >= 40:
            return "优秀"
        elif snr >= 30:
            return "良好"
        elif snr >= 20:
            return "一般"
        else:
            return "较差"

    def _setup_chinese_font():
        """设置中文字体"""
        font_options = [
            'Microsoft YaHei',
            'SimHei',
            'STHeiti',
            'PingFang SC',
            'Noto Sans CJK SC',
            'WenQuanYi Micro Hei',
            'DejaVu Sans'
        ]

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = font_options
        plt.rcParams['axes.unicode_minus'] = False

    def _create_visualization(snr_values: List[float], filenames: List[str],
                              indices: List[int], total_files: int):
        """创建可视化图表"""
        _setup_chinese_font()

        n_files = len(snr_values)

        # 动态调整图表大小
        fig_width = min(24, max(14, n_files * 0.25))
        fig_height = 10

        fig, (ax_main, ax_bar) = plt.subplots(2, 1, figsize=(fig_width, fig_height),
                                              height_ratios=[2, 1])
        fig.suptitle('AI翻唱音频质量评估 - 信噪比(SNR)分析',
                     fontsize=16, fontweight='bold', y=0.98)

        x = np.arange(n_files)
        colors = [_get_snr_color(snr) for snr in snr_values]

        # ========== 上图：趋势折线图 ==========
        # 添加阈值区域
        y_min = max(0, min(snr_values) - 5)
        y_max = max(snr_values) + 5

        ax_main.fill_between([x[0] - 0.5, x[-1] + 0.5], 40, y_max, alpha=0.15,
                             color='green', label='优秀区 (>40dB): 无明显噪声')
        ax_main.fill_between([x[0] - 0.5, x[-1] + 0.5], 30, 40, alpha=0.15,
                             color='orange', label='良好区 (30-40dB): 轻微底噪')
        ax_main.fill_between([x[0] - 0.5, x[-1] + 0.5], 20, 30, alpha=0.15,
                             color='gold', label='一般区 (20-30dB): 可感知噪声')
        ax_main.fill_between([x[0] - 0.5, x[-1] + 0.5], y_min, 20, alpha=0.15,
                             color='red', label='较差区 (<20dB): 明显干扰')

        # 阈值线
        for threshold, color, style in [(40, 'green', '--'),
                                        (30, 'orange', '--'),
                                        (20, 'red', '--')]:
            if y_min < threshold < y_max:
                ax_main.axhline(y=threshold, color=color, linestyle=style,
                                linewidth=1.5, alpha=0.8)

        # 折线和散点
        ax_main.plot(x, snr_values, 'b-', linewidth=1.5, alpha=0.6, zorder=2)
        ax_main.scatter(x, snr_values, c=colors, s=60, edgecolors='black',
                        linewidth=0.5, zorder=3)

        # 标注最大最小值
        max_idx = np.argmax(snr_values)
        min_idx = np.argmin(snr_values)

        ax_main.annotate(f'最高: {snr_values[max_idx]:.1f}dB\n{filenames[max_idx]}',
                         xy=(max_idx, snr_values[max_idx]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=8, ha='left',
                         arrowprops=dict(arrowstyle='->', color='green', lw=1),
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.8, edgecolor='green'))

        if min_idx != max_idx:
            ax_main.annotate(f'最低: {snr_values[min_idx]:.1f}dB\n{filenames[min_idx]}',
                             xy=(min_idx, snr_values[min_idx]),
                             xytext=(10, -25), textcoords='offset points',
                             fontsize=8, ha='left',
                             arrowprops=dict(arrowstyle='->', color='red', lw=1),
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       alpha=0.8, edgecolor='red'))

        # 动态调整Y轴以突出差异
        snr_range = max(snr_values) - min(snr_values)
        if snr_range < 3:
            y_center = np.mean(snr_values)
            ax_main.set_ylim(y_center - 8, y_center + 8)
        else:
            margin = max(snr_range * 0.15, 2)
            ax_main.set_ylim(min(snr_values) - margin, max(snr_values) + margin)

        ax_main.set_xlim(-0.5, n_files - 0.5)
        ax_main.set_ylabel('信噪比 SNR (dB)', fontsize=11)
        ax_main.set_xlabel('')
        ax_main.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax_main.grid(True, alpha=0.3, linestyle=':')

        # X轴标签（简化显示）
        if n_files <= 25:
            ax_main.set_xticks(x)
            ax_main.set_xticklabels([f'{i + 1}' for i in range(n_files)], fontsize=8)
        else:
            step = max(1, n_files // 20)
            tick_pos = list(range(0, n_files, step))
            if n_files - 1 not in tick_pos:
                tick_pos.append(n_files - 1)
            ax_main.set_xticks(tick_pos)
            ax_main.set_xticklabels([f'{i + 1}' for i in tick_pos], fontsize=8)

        # ========== 下图：柱状图 ==========
        bars = ax_bar.bar(x, snr_values, color=colors, edgecolor='black', linewidth=0.5)

        # 阈值线
        for threshold, color in [(40, 'green'), (30, 'orange'), (20, 'red')]:
            ax_bar.axhline(y=threshold, color=color, linestyle='--',
                           linewidth=1.2, alpha=0.7)

        # Y轴范围
        ax_bar.set_ylim(0, max(snr_values) * 1.15)
        ax_bar.set_xlim(-0.5, n_files - 0.5)

        # X轴文件名标签
        if n_files <= 15:
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(filenames, rotation=45, ha='right', fontsize=8)
        elif n_files <= 40:
            step = max(1, n_files // 12)
            tick_pos = list(range(0, n_files, step))
            ax_bar.set_xticks(tick_pos)
            ax_bar.set_xticklabels([filenames[i] for i in tick_pos],
                                   rotation=45, ha='right', fontsize=7)
        else:
            step = max(1, n_files // 15)
            tick_pos = list(range(0, n_files, step))
            ax_bar.set_xticks(tick_pos)
            short_names = [os.path.splitext(filenames[i])[0][-15:] for i in tick_pos]
            ax_bar.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)

        ax_bar.set_ylabel('SNR (dB)', fontsize=10)
        ax_bar.set_xlabel('文件（按模型训练轮数递增 →）', fontsize=10)
        ax_bar.grid(True, alpha=0.3, linestyle=':', axis='y')

        # ========== 统计信息文本框 ==========
        snr_arr = np.array(snr_values)
        excellent = np.sum(snr_arr >= 40)
        good = np.sum((snr_arr >= 30) & (snr_arr < 40))
        fair = np.sum((snr_arr >= 20) & (snr_arr < 30))
        poor = np.sum(snr_arr < 20)

        stats_text = (
            f"◆ 信噪比(SNR)说明\n"
            f"  公式: SNR = 10×log₁₀(信号能量/噪声能量)\n"
            f"  方法: 通过静音段检测估算底噪水平\n\n"
            f"◆ 质量阈值参考\n"
            f"  >40dB: 优秀(无噪) | 30-40dB: 良好(轻微)\n"
            f"  20-30dB: 一般(可感知) | <20dB: 较差(干扰)\n\n"
            f"◆ 当前数据统计 (共{n_files}个文件)\n"
            f"  范围: {min(snr_values):.1f} ~ {max(snr_values):.1f} dB\n"
            f"  均值: {np.mean(snr_values):.1f} dB | 中位数: {np.median(snr_values):.1f} dB\n"
            f"  分布: 优秀{excellent} | 良好{good} | 一般{fair} | 较差{poor}"
        )

        fig.text(0.99, 0.50, stats_text, transform=fig.transFigure,
                 fontsize=9, verticalalignment='center', horizontalalignment='right',
                 family='sans-serif',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                           alpha=0.85, edgecolor='gray', linewidth=1))

        plt.tight_layout(rect=[0, 0, 0.78, 0.95])
        plt.show()

    # ==================== 主处理逻辑 ====================

    if not wav_paths:
        print("错误: 未提供WAV文件路径")
        return {}

    print(f"开始处理 {len(wav_paths)} 个WAV文件...")

    # 并发处理
    results = [None] * len(wav_paths)
    max_workers = min(os.cpu_count() or 4, 8, len(wav_paths))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_calculate_snr_single, path): idx
                         for idx, path in enumerate(wav_paths)}

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"  进度: {completed}/{len(wav_paths)}")

    # 提取有效结果
    valid_data = [(i, r) for i, r in enumerate(results) if r and r['snr'] is not None]

    # 报告错误
    errors = [(i, r) for i, r in enumerate(results) if r and r['error'] is not None]
    if errors:
        print(f"\n⚠ {len(errors)} 个文件处理失败:")
        for idx, r in errors[:5]:
            print(f"  - {r['filename']}: {r['error']}")
        if len(errors) > 5:
            print(f"  ... 等共 {len(errors)} 个")

    if not valid_data:
        print("错误: 没有成功处理的文件！")
        return {}

    # 准备可视化数据
    indices = [d[0] for d in valid_data]
    snr_values = [d[1]['snr'] for d in valid_data]
    filenames = [d[1]['filename'] for d in valid_data]

    print(f"\n✓ 成功处理 {len(valid_data)} 个文件")
    print(f"  SNR范围: {min(snr_values):.1f} ~ {max(snr_values):.1f} dB")
    print(f"  平均SNR: {np.mean(snr_values):.1f} dB")

    # 可视化
    _create_visualization(snr_values, filenames, indices, len(wav_paths))

    # 返回结果字典
    return {r['filename']: r['snr'] for r in results if r and r['snr'] is not None}


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import glob

    # 示例：获取某目录下所有wav文件
    # wav_files = sorted(glob.glob("/path/to/your/wav/files/*.wav"))

    # 或手动指定文件列表
    wav_files = [
        "model_epoch_100.wav",
        "model_epoch_200.wav",
        "model_epoch_300.wav",
        # ... 更多文件
    ]

    # 调用分析函数
    # results = analyze_audio_snr(wav_files)

    # 查看结果
    # for filename, snr in results.items():
    #     print(f"{filename}: {snr:.1f} dB")