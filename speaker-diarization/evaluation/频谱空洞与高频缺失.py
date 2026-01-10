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

+ 频谱空洞与高频缺失（Spectrogram Analysis）
    + 含义：检查10kHz以上高频分布；AI低采样率模型常见截断。
    + 16–20kHz合理能量：高质量
    + 截断：低质量模型
"""

"""
AI翻唱音频高频质量分析工具
分析指标：频谱空洞与高频缺失（Spectrogram Analysis）

依赖安装：
pip install numpy scipy matplotlib

可选依赖（更好的音频支持）：
pip install soundfile
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def analyze_high_frequency_quality(
        wav_paths: List[str],
        max_workers: int = 8,
        show_table: bool = True
) -> Dict:
    """
    分析AI翻唱wav文件的高频质量（频谱空洞与高频缺失）

    参数:
        wav_paths: wav文件路径列表（按模型轮数递增排序）
        max_workers: 并发处理的最大线程数
        show_table: 是否显示详情表格窗口

    返回:
        包含分析结果的字典
    """

    # ==================== 内部函数定义 ====================

    def read_audio(wav_path: str) -> Tuple[int, np.ndarray]:
        """读取音频文件，支持多种格式"""
        try:
            # 优先使用soundfile（支持更多格式）
            import soundfile as sf
            audio, sample_rate = sf.read(wav_path)
            return sample_rate, audio
        except ImportError:
            # 回退到scipy
            sample_rate, audio = wavfile.read(wav_path)
            return sample_rate, audio

    def analyze_single_file(wav_path: str) -> Dict:
        """分析单个wav文件的频谱特征"""
        try:
            # 读取音频文件
            sample_rate, audio = read_audio(wav_path)

            # 转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # 归一化到浮点数
            audio = audio.astype(np.float64)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            # 计算短时傅里叶变换（STFT）
            nperseg = min(4096, len(audio) // 4)
            frequencies, times, Sxx = signal.spectrogram(
                audio,
                fs=sample_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2,
                nfft=nperseg * 2,
                scaling='spectrum'
            )

            # 转换为dB，避免log(0)
            Sxx_db = 10 * np.log10(Sxx + 1e-12)

            # 计算平均频谱
            mean_spectrum = np.mean(Sxx_db, axis=1)

            # 定义频段掩码
            low_mask = frequencies < 10000  # 0-10kHz
            mid_high_mask = (frequencies >= 10000) & (frequencies < 16000)  # 10-16kHz
            high_mask = (frequencies >= 16000) & (frequencies <= 20000)  # 16-20kHz
            ultra_high_mask = (frequencies > 20000) & (frequencies <= sample_rate / 2)  # >20kHz

            # 计算各频段平均能量
            low_energy = np.mean(mean_spectrum[low_mask]) if np.any(low_mask) else -100
            mid_high_energy = np.mean(mean_spectrum[mid_high_mask]) if np.any(mid_high_mask) else -100
            high_energy = np.mean(mean_spectrum[high_mask]) if np.any(high_mask) else -100
            ultra_high_energy = np.mean(mean_spectrum[ultra_high_mask]) if np.any(ultra_high_mask) else -100

            # 计算高频能量比（相对于低频的dB差值）
            high_freq_ratio = high_energy - low_energy
            mid_high_ratio = mid_high_energy - low_energy

            # 检测频谱截断频率
            noise_floor = np.percentile(mean_spectrum, 5)
            threshold = noise_floor + 10  # 高于噪声底10dB认为有效
            cutoff_freq = 0

            for i in range(len(frequencies) - 1, -1, -1):
                if mean_spectrum[i] > threshold:
                    cutoff_freq = frequencies[i]
                    break

            # 计算10kHz以上的能量占比
            total_energy_linear = np.mean(10 ** (Sxx_db / 10))
            high_freq_energy_linear = np.mean(10 ** (Sxx_db[frequencies >= 10000] / 10)) if np.any(
                frequencies >= 10000) else 0
            high_freq_percentage = (
                        high_freq_energy_linear / total_energy_linear * 100) if total_energy_linear > 0 else 0

            # 计算高频平滑度（能量变化的标准差，越小越平滑）
            if np.any(high_mask):
                high_freq_smoothness = np.std(np.diff(mean_spectrum[high_mask]))
            else:
                high_freq_smoothness = float('inf')

            # 检测频谱空洞（能量突然下降的区域）
            spectrum_diff = np.diff(mean_spectrum)
            holes = np.where(spectrum_diff < -15)[0]  # 能量下降超过15dB
            hole_count = len(holes)

            return {
                'path': wav_path,
                'filename': os.path.basename(wav_path),
                'sample_rate': sample_rate,
                'duration': len(audio) / sample_rate,
                'low_energy': float(low_energy),
                'mid_high_energy': float(mid_high_energy),
                'high_energy': float(high_energy),
                'ultra_high_energy': float(ultra_high_energy),
                'high_freq_ratio': float(high_freq_ratio),
                'mid_high_ratio': float(mid_high_ratio),
                'cutoff_freq': float(cutoff_freq),
                'high_freq_percentage': float(high_freq_percentage),
                'high_freq_smoothness': float(high_freq_smoothness),
                'hole_count': int(hole_count),
                'mean_spectrum': mean_spectrum,
                'frequencies': frequencies,
                'success': True,
                'error': None
            }

        except Exception as e:
            return {
                'path': wav_path,
                'filename': os.path.basename(wav_path),
                'success': False,
                'error': str(e)
            }

    def normalize_score(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
        """归一化评分到0-100"""
        if len(values) == 0:
            return np.array([])
        v_min, v_max = np.min(values), np.max(values)
        if v_max == v_min:
            return np.ones_like(values) * 50.0
        normalized = (values - v_min) / (v_max - v_min)
        if not higher_is_better:
            normalized = 1 - normalized
        return normalized * 100

    def setup_chinese_font():
        """设置中文字体"""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'DejaVu Sans',
            'Arial Unicode MS', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
        ]
        plt.rcParams['axes.unicode_minus'] = False

    # ==================== 主处理逻辑 ====================

    print("=" * 60)
    print("【频谱空洞与高频缺失分析】")
    print("=" * 60)
    print(f"待分析文件数: {len(wav_paths)}")
    print(f"并发线程数: {max_workers}")
    print("正在分析中...")

    # 并发处理所有文件
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(analyze_single_file, path): path
            for path in wav_paths
        }

        completed = 0
        for future in as_completed(future_to_path):
            results.append(future.result())
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"  进度: {completed}/{len(wav_paths)}")

    # 按原始顺序排序
    path_to_result = {r['path']: r for r in results}
    ordered_results = [path_to_result[path] for path in wav_paths if path in path_to_result]

    # 分离成功和失败的结果
    successful_results = [r for r in ordered_results if r['success']]
    failed_results = [r for r in ordered_results if not r['success']]

    if failed_results:
        print(f"\n⚠ 警告: {len(failed_results)} 个文件处理失败:")
        for r in failed_results[:5]:  # 只显示前5个
            print(f"   - {r['filename']}: {r['error']}")
        if len(failed_results) > 5:
            print(f"   ... 还有 {len(failed_results) - 5} 个文件")

    if not successful_results:
        print("\n✗ 错误: 没有成功处理的文件!")
        return {'success': False, 'error': '没有成功处理的文件'}

    print(f"\n✓ 成功分析 {len(successful_results)} 个文件")

    # ==================== 数据准备 ====================

    n_files = len(successful_results)
    x_indices = np.arange(n_files)

    # 提取各项指标
    filenames = [r['filename'] for r in successful_results]
    cutoff_freqs = np.array([r['cutoff_freq'] / 1000 for r in successful_results])  # kHz
    high_energies = np.array([r['high_energy'] for r in successful_results])
    mid_high_energies = np.array([r['mid_high_energy'] for r in successful_results])
    high_freq_ratios = np.array([r['high_freq_ratio'] for r in successful_results])
    high_freq_percentages = np.array([r['high_freq_percentage'] for r in successful_results])
    hole_counts = np.array([r['hole_count'] for r in successful_results])

    # 计算各项评分
    cutoff_scores = normalize_score(cutoff_freqs, higher_is_better=True)
    high_energy_scores = normalize_score(high_energies, higher_is_better=True)
    ratio_scores = normalize_score(high_freq_ratios, higher_is_better=True)
    hole_scores = normalize_score(hole_counts, higher_is_better=False)

    # 综合评分（加权平均）
    overall_scores = (
            cutoff_scores * 0.35 +
            high_energy_scores * 0.25 +
            ratio_scores * 0.25 +
            hole_scores * 0.15
    )

    # ==================== 可视化 ====================

    setup_chinese_font()

    # 创建主图表窗口
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        '频谱空洞与高频缺失分析\n(Spectrogram Analysis - High Frequency Quality)',
        fontsize=14, fontweight='bold', y=0.98
    )

    # 根据文件数量决定X轴显示策略
    if n_files <= 30:
        x_labels = [f"{i + 1}" for i in range(n_files)]
        x_ticks = x_indices
        rotation = 0
    elif n_files <= 60:
        step = 2
        x_labels = [f"{i + 1}" if i % step == 0 else "" for i in range(n_files)]
        x_ticks = x_indices
        rotation = 45
    else:
        step = max(1, n_files // 25)
        x_ticks = x_indices[::step]
        x_labels = [f"{i + 1}" for i in range(0, n_files, step)]
        rotation = 45

    # ---------- 子图1: 频谱截断频率趋势 ----------
    ax1 = fig.add_subplot(2, 2, 1)

    # 绘制数据点和趋势线
    ax1.plot(x_indices, cutoff_freqs, 'o-', color='#2E86AB',
             markersize=4, linewidth=1.5, label='截断频率', alpha=0.8)

    # 添加阈值参考线
    ax1.axhline(y=20, color='#28A745', linestyle='--', linewidth=2,
                label='理想阈值 (20kHz)', alpha=0.8)
    ax1.axhline(y=16, color='#FFC107', linestyle='--', linewidth=2,
                label='良好阈值 (16kHz)', alpha=0.8)
    ax1.axhline(y=10, color='#DC3545', linestyle='--', linewidth=2,
                label='警戒阈值 (10kHz)', alpha=0.8)

    # 填充质量区域
    ax1.axhspan(16, 24, alpha=0.1, color='green', label='_高质量区域')
    ax1.axhspan(10, 16, alpha=0.1, color='yellow', label='_中等区域')
    ax1.axhspan(0, 10, alpha=0.1, color='red', label='_低质量区域')

    # 动态调整Y轴以突出差异
    y_min, y_max = np.min(cutoff_freqs), np.max(cutoff_freqs)
    y_range = y_max - y_min
    y_margin = max(y_range * 0.2, 1)
    ax1.set_ylim(max(0, y_min - y_margin), min(24, y_max + y_margin + 2))

    ax1.set_xticks(x_ticks if n_files > 30 else x_indices)
    ax1.set_xticklabels(x_labels if n_files <= 60 else [f"{i + 1}" for i in range(0, n_files, step)],
                        rotation=rotation, fontsize=8)
    ax1.set_xlabel('文件序号 (按训练轮数递增)', fontsize=10)
    ax1.set_ylabel('截断频率 (kHz)', fontsize=10)
    ax1.set_title('① 频谱截断频率趋势', fontsize=11, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', framealpha=0, fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # 指标说明（透明背景）
    desc1 = """【指标说明】
检测音频有效频率的上限
• ≥20kHz: ★★★ 理想（绿线）
• 16-20kHz: ★★ 良好（黄线）
• 10-16kHz: ★ 一般
• <10kHz: ✗ 严重截断（红线）

AI低采样率模型常见截断问题"""
    ax1.text(0.98, 0.02, desc1, transform=ax1.transAxes, fontsize=7,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='none',
                       edgecolor='gray', alpha=0.5))

    # ---------- 子图2: 高频能量分布对比 ----------
    ax2 = fig.add_subplot(2, 2, 2)

    bar_width = 0.35
    bars1 = ax2.bar(x_indices - bar_width / 2, mid_high_energies, bar_width,
                    label='10-16kHz 能量', color='#4ECDC4', alpha=0.85, edgecolor='white')
    bars2 = ax2.bar(x_indices + bar_width / 2, high_energies, bar_width,
                    label='16-20kHz 能量', color='#FF6B6B', alpha=0.85, edgecolor='white')

    # 动态调整Y轴
    all_energies = np.concatenate([mid_high_energies, high_energies])
    y_min, y_max = np.min(all_energies), np.max(all_energies)
    y_range = y_max - y_min
    y_margin = max(y_range * 0.15, 3)
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)

    ax2.set_xticks(x_ticks if n_files > 30 else x_indices)
    ax2.set_xticklabels(x_labels if n_files <= 60 else [f"{i + 1}" for i in range(0, n_files, step)],
                        rotation=rotation, fontsize=8)
    ax2.set_xlabel('文件序号 (按训练轮数递增)', fontsize=10)
    ax2.set_ylabel('平均能量 (dB)', fontsize=10)
    ax2.set_title('② 高频段能量分布', fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', framealpha=0, fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle=':', axis='y')

    desc2 = """【指标说明】
检测高频成分的丰富程度
• 16-20kHz能量越高越好
• 两频段能量接近=频谱平滑
• 16-20kHz急剧下降=模型截断

高质量AI模型应保持
高频能量的平稳分布"""
    ax2.text(0.98, 0.02, desc2, transform=ax2.transAxes, fontsize=7,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='none',
                       edgecolor='gray', alpha=0.5))

    # ---------- 子图3: 高低频能量比趋势 ----------
    ax3 = fig.add_subplot(2, 2, 3)

    # 使用面积图增强可读性
    ax3.fill_between(x_indices, high_freq_ratios, alpha=0.3, color='#9B59B6')
    ax3.plot(x_indices, high_freq_ratios, 'o-', color='#9B59B6',
             linewidth=2, markersize=4, label='高低频能量比')

    # 添加阈值参考线
    ax3.axhline(y=-15, color='#28A745', linestyle='--', linewidth=2,
                label='优秀阈值 (-15dB)', alpha=0.8)
    ax3.axhline(y=-25, color='#FFC107', linestyle='--', linewidth=2,
                label='良好阈值 (-25dB)', alpha=0.8)
    ax3.axhline(y=-35, color='#DC3545', linestyle='--', linewidth=2,
                label='警戒阈值 (-35dB)', alpha=0.8)

    # 动态调整Y轴
    y_min, y_max = np.min(high_freq_ratios), np.max(high_freq_ratios)
    y_range = y_max - y_min
    y_margin = max(y_range * 0.2, 3)
    ax3.set_ylim(y_min - y_margin, y_max + y_margin)

    ax3.set_xticks(x_ticks if n_files > 30 else x_indices)
    ax3.set_xticklabels(x_labels if n_files <= 60 else [f"{i + 1}" for i in range(0, n_files, step)],
                        rotation=rotation, fontsize=8)
    ax3.set_xlabel('文件序号 (按训练轮数递增)', fontsize=10)
    ax3.set_ylabel('能量比 (dB)', fontsize=10)
    ax3.set_title('③ 高频/低频能量比 (16-20kHz vs 0-10kHz)', fontsize=11, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', framealpha=0, fontsize=8)
    ax3.grid(True, alpha=0.3, linestyle=':')

    desc3 = """【指标说明】
衡量高频相对强度（dB差值）
• ≥-15dB: ★★★ 高频丰富
• -15~-25dB: ★★ 正常范围
• -25~-35dB: ★ 高频偏弱
• <-35dB: ✗ 高频严重缺失

值越接近0，高频越丰富"""
    ax3.text(0.98, 0.02, desc3, transform=ax3.transAxes, fontsize=7,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='none',
                       edgecolor='gray', alpha=0.5))

    # ---------- 子图4: 综合质量评分热力图 ----------
    ax4 = fig.add_subplot(2, 2, 4)

    # 构建评分矩阵
    score_matrix = np.array([
        cutoff_scores,
        high_energy_scores,
        ratio_scores,
        hole_scores,
        overall_scores
    ])

    # 绘制热力图
    im = ax4.imshow(score_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)

    # Y轴标签
    y_labels = ['截断频率\n(40%)', '高频能量\n(25%)', '能量比\n(25%)',
                '频谱空洞\n(15%)', '【综合评分】']
    ax4.set_yticks(np.arange(5))
    ax4.set_yticklabels(y_labels, fontsize=9)

    # X轴标签
    if n_files <= 40:
        ax4.set_xticks(x_indices)
        ax4.set_xticklabels([f"{i + 1}" for i in range(n_files)], fontsize=7, rotation=45)
    else:
        step = max(1, n_files // 30)
        ax4.set_xticks(x_indices[::step])
        ax4.set_xticklabels([f"{i + 1}" for i in range(0, n_files, step)], fontsize=8)

    ax4.set_xlabel('文件序号 (按训练轮数递增)', fontsize=10)
    ax4.set_title('④ 各指标质量评分热力图 (0-100分)', fontsize=11, fontweight='bold', pad=10)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8, pad=0.02)
    cbar.set_label('质量评分 (分)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # 在热力图上标注最高/最低分
    best_idx = np.argmax(overall_scores)
    worst_idx = np.argmin(overall_scores)
    ax4.axvline(x=best_idx, color='blue', linestyle='-', linewidth=2, alpha=0.7)
    ax4.axvline(x=worst_idx, color='red', linestyle='-', linewidth=2, alpha=0.7)

    desc4 = f"""【评分说明】
颜色: 绿=高分 | 红=低分

权重分配:
• 截断频率: 40%
• 高频能量: 25%
• 能量比: 25%
• 频谱空洞: 15%

蓝线=最佳 (#{best_idx + 1})
红线=最差 (#{worst_idx + 1})"""
    ax4.text(1.18, 0.5, desc4, transform=ax4.transAxes, fontsize=7,
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='none',
                       edgecolor='gray', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ==================== 详情表格窗口 ====================

    if show_table and n_files <= 100:
        fig2 = plt.figure(figsize=(16, max(6, n_files * 0.18 + 2)))
        fig2.suptitle('文件详情列表', fontsize=12, fontweight='bold', y=0.98)

        ax_table = fig2.add_subplot(1, 1, 1)
        ax_table.axis('off')

        # 准备表格数据
        headers = ['序号', '文件名', '采样率', '截断频率\n(kHz)',
                   '16-20kHz\n能量(dB)', '高低频比\n(dB)', '频谱空洞\n(个)', '综合评分']

        table_data = []
        for i, r in enumerate(successful_results):
            short_name = r['filename']
            if len(short_name) > 35:
                short_name = short_name[:15] + '...' + short_name[-15:]

            table_data.append([
                str(i + 1),
                short_name,
                f"{r['sample_rate'] // 1000}kHz",
                f"{r['cutoff_freq'] / 1000:.1f}",
                f"{r['high_energy']:.1f}",
                f"{r['high_freq_ratio']:.1f}",
                str(r['hole_count']),
                f"{overall_scores[i]:.1f}"
            ])

        # 创建表格
        table = ax_table.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colWidths=[0.05, 0.25, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.4)

        # 设置表头样式
        for j, header in enumerate(headers):
            table[(0, j)].set_facecolor('#4A90D9')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # 根据评分设置单元格颜色
        for i in range(len(successful_results)):
            score = overall_scores[i]
            if score >= 75:
                color = '#C8E6C9'  # 浅绿
            elif score >= 50:
                color = '#FFF9C4'  # 浅黄
            elif score >= 25:
                color = '#FFE0B2'  # 浅橙
            else:
                color = '#FFCDD2'  # 浅红

            # 给综合评分列着色
            table[(i + 1, 7)].set_facecolor(color)

            # 交替行背景色
            if i % 2 == 1:
                for j in range(7):
                    current_color = table[(i + 1, j)].get_facecolor()
                    if current_color == (1, 1, 1, 1):  # 白色
                        table[(i + 1, j)].set_facecolor('#F5F5F5')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ==================== 显示图表 ====================

    plt.show()

    # ==================== 打印总结 ====================

    print("\n" + "=" * 60)
    print("【分析结果总结】")
    print("=" * 60)
    print(f"  分析文件数: {n_files}")
    print(f"  平均截断频率: {np.mean(cutoff_freqs):.2f} kHz")
    print(f"  平均16-20kHz能量: {np.mean(high_energies):.1f} dB")
    print(f"  平均高低频比: {np.mean(high_freq_ratios):.1f} dB")
    print(f"  平均综合评分: {np.mean(overall_scores):.1f} 分")
    print("-" * 60)
    print(
        f"  🏆 最佳文件: #{np.argmax(overall_scores) + 1} - {successful_results[np.argmax(overall_scores)]['filename']}")
    print(f"      综合评分: {np.max(overall_scores):.1f} 分")
    print(
        f"  ⚠ 最差文件: #{np.argmin(overall_scores) + 1} - {successful_results[np.argmin(overall_scores)]['filename']}")
    print(f"      综合评分: {np.min(overall_scores):.1f} 分")
    print("=" * 60)

    # 返回结果字典
    return {
        'success': True,
        'n_files': n_files,
        'results': successful_results,
        'scores': {
            'cutoff_scores': cutoff_scores.tolist(),
            'high_energy_scores': high_energy_scores.tolist(),
            'ratio_scores': ratio_scores.tolist(),
            'hole_scores': hole_scores.tolist(),
            'overall_scores': overall_scores.tolist()
        },
        'best_file': {
            'index': int(np.argmax(overall_scores)),
            'filename': successful_results[np.argmax(overall_scores)]['filename'],
            'score': float(np.max(overall_scores))
        },
        'worst_file': {
            'index': int(np.argmin(overall_scores)),
            'filename': successful_results[np.argmin(overall_scores)]['filename'],
            'score': float(np.min(overall_scores))
        }
    }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import glob

    # 示例：获取某目录下所有wav文件
    # wav_files = sorted(glob.glob("path/to/your/wav/files/*.wav"))

    # 或者手动指定文件列表
    wav_files = [
        r"D:\ai_cover\model_epoch_100.wav",
        r"D:\ai_cover\model_epoch_200.wav",
        r"D:\ai_cover\model_epoch_300.wav",
        # ... 添加更多文件路径
    ]

    # 调用分析函数
    if wav_files:
        results = analyze_high_frequency_quality(
            wav_paths=wav_files,
            max_workers=8,  # 并发线程数
            show_table=True  # 是否显示详情表格
        )