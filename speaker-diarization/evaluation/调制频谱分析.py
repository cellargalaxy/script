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

+ 调制频谱分析
    + 含义：3–8Hz振幅调制（颤音范围）；检测自然颤音。
    + 强度-20至-30dB，频率4–6Hz：健康
    + 偏离：异常
"""

# pip install numpy scipy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re


def analyze_modulation_spectrum(wav_paths):
    """
    对AI翻唱WAV文件进行调制频谱分析

    分析3-8Hz振幅调制范围（颤音范围），检测自然颤音特征。
    健康标准：强度-20至-30dB，频率4-6Hz
    偏离此范围表示颤音异常

    参数:
        wav_paths: list[str] - WAV文件路径的字符串数组（已按模型轮数递增排序）

    返回:
        list[dict] - 各文件的分析结果
    """

    if not wav_paths:
        print("错误: 文件路径数组为空")
        return []

    # ==================== 特征提取函数 ====================
    def extract_modulation_features(wav_path):
        """提取单个WAV文件的调制频谱特征（线程安全）"""
        try:
            # 读取WAV文件
            sr, audio = wavfile.read(wav_path)

            # 转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # 转换为浮点并归一化
            audio = audio.astype(np.float64)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            # 使用希尔伯特变换提取包络
            analytic_signal = hilbert(audio)
            envelope = np.abs(analytic_signal)

            # 移除直流分量
            envelope = envelope - np.mean(envelope)

            # 对包络进行低通滤波（截止频率15Hz，保留颤音范围）
            nyq = sr / 2.0
            cutoff = 15.0
            if cutoff < nyq:
                b, a = butter(4, cutoff / nyq, btype='low')
                envelope = filtfilt(b, a, envelope)

            # 下采样到约100Hz进行低频FFT分析
            target_sr = 100
            downsample_factor = max(1, int(sr / target_sr))
            envelope_ds = envelope[::downsample_factor]
            sr_env = sr / downsample_factor

            # FFT分析
            n = len(envelope_ds)
            fft_result = np.fft.rfft(envelope_ds)
            freqs = np.fft.rfftfreq(n, 1.0 / sr_env)
            power_spectrum = np.abs(fft_result) ** 2

            # 3-8Hz颤音范围分析
            vibrato_mask = (freqs >= 3) & (freqs <= 8)
            vibrato_freqs = freqs[vibrato_mask]
            vibrato_power = power_spectrum[vibrato_mask]

            # 计算总功率（排除极低频）
            total_power = np.sum(power_spectrum[freqs > 0.5])

            if len(vibrato_power) > 0 and total_power > 0:
                # 主导调制频率（3-8Hz范围内功率最大的频率）
                max_idx = np.argmax(vibrato_power)
                peak_freq = vibrato_freqs[max_idx]

                # 颤音范围能量占比（相对于总能量）
                vibrato_energy = np.sum(vibrato_power)
                vibrato_ratio = vibrato_energy / total_power
                vibrato_ratio_db = 10 * np.log10(vibrato_ratio + 1e-12)

                # 4-6Hz健康范围在颤音范围中的占比
                healthy_mask = (vibrato_freqs >= 4) & (vibrato_freqs <= 6)
                healthy_power = vibrato_power[healthy_mask]
                if vibrato_energy > 0 and len(healthy_power) > 0:
                    healthy_ratio = np.sum(healthy_power) / vibrato_energy
                else:
                    healthy_ratio = 0.0
            else:
                peak_freq = 0.0
                vibrato_ratio_db = -60.0
                healthy_ratio = 0.0

            return {
                'path': wav_path,
                'filename': os.path.basename(wav_path),
                'peak_freq': float(peak_freq),
                'vibrato_ratio_db': float(vibrato_ratio_db),
                'healthy_ratio': float(healthy_ratio),
                'success': True,
                'error': None
            }

        except Exception as e:
            return {
                'path': wav_path,
                'filename': os.path.basename(wav_path),
                'peak_freq': 0.0,
                'vibrato_ratio_db': -60.0,
                'healthy_ratio': 0.0,
                'success': False,
                'error': str(e)
            }

    # ==================== 并发处理 ====================
    print(f"开始处理 {len(wav_paths)} 个文件...")

    results = [None] * len(wav_paths)
    max_workers = min(os.cpu_count() or 4, len(wav_paths), 16)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(extract_modulation_features, path): idx
            for idx, path in enumerate(wav_paths)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"进度: {completed}/{len(wav_paths)}")

    # 统计结果
    valid_results = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if failed:
        print(f"\n警告: {len(failed)} 个文件处理失败:")
        for r in failed[:5]:
            print(f"  - {r['filename']}: {r['error']}")
        if len(failed) > 5:
            print(f"  ... 还有 {len(failed) - 5} 个失败文件")

    if not valid_results:
        print("\n错误: 没有成功处理的文件，无法生成图表")
        return results

    print(f"\n成功处理 {len(valid_results)} 个文件，正在生成图表...")

    # ==================== 可视化设置 ====================
    n_files = len(valid_results)

    # 设置中文字体（非等宽字体）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC',
        'Hiragino Sans GB', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'DejaVu Sans', 'Arial'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    # 提取数据
    filenames = [r['filename'] for r in valid_results]
    peak_freqs = np.array([r['peak_freq'] for r in valid_results])
    vibrato_dbs = np.array([r['vibrato_ratio_db'] for r in valid_results])
    healthy_ratios = np.array([r['healthy_ratio'] for r in valid_results])

    # 生成简化标签
    short_names = []
    for i, f in enumerate(filenames):
        name = os.path.splitext(f)[0]
        if len(name) > 15:
            nums = re.findall(r'\d+', name)
            short_names.append(f"#{nums[-1]}" if nums else f"#{i + 1}")
        else:
            short_names.append(name)

    # 计算图表尺寸
    fig_width = max(14, min(n_files * 0.25 + 2, 28))
    fig_height = 15

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle('调制频谱分析 - AI翻唱质量评价', fontsize=14, fontweight='bold', y=0.98)

    x = np.arange(n_files)

    # X轴标签显示策略
    if n_files <= 30:
        tick_step = 1
    elif n_files <= 60:
        tick_step = 2
    else:
        tick_step = max(1, n_files // 25)

    tick_positions = x[::tick_step]
    tick_labels = [short_names[i] for i in range(0, n_files, tick_step)]

    # ==================== 子图1: 颤音主频率 ====================
    ax1 = fig.add_subplot(3, 1, 1)

    # 健康范围背景（4-6Hz）
    ax1.fill_between([-0.5, n_files - 0.5], 4, 6,
                     color='#90EE90', alpha=0.3, label='健康范围 (4-6 Hz)')

    # 数据折线
    ax1.plot(x, peak_freqs, 'o-', color='#4682B4', linewidth=1.5,
             markersize=3 if n_files > 50 else 5, label='主导调制频率', alpha=0.8)

    # 趋势线
    if n_files >= 3:
        z1 = np.polyfit(x, peak_freqs, 1)
        p1 = np.poly1d(z1)
        trend_dir = "↑" if z1[0] > 0 else "↓"
        trend_label = f'趋势线 ({trend_dir} {abs(z1[0]):.4f} Hz/轮)'
        ax1.plot(x, p1(x), '--', color='#191970', linewidth=2,
                 alpha=0.7, label=trend_label)

    # 边界线
    ax1.axhline(y=4, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=6, color='green', linestyle=':', linewidth=1.5, alpha=0.7)

    ax1.set_ylabel('主导调制频率 (Hz)', fontsize=11)
    ax1.set_title('【颤音频率】3-8Hz范围内的主导调制频率 | 阈值: 4-6Hz为健康',
                  fontsize=11, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.set_xlim(-0.5, n_files - 0.5)
    ax1.grid(True, alpha=0.3, linestyle='-')

    # 动态Y轴（放大差异）
    freq_min, freq_max = peak_freqs.min(), peak_freqs.max()
    freq_range = freq_max - freq_min
    if freq_range < 1:
        freq_center = (freq_min + freq_max) / 2
        ax1.set_ylim(freq_center - 1.5, freq_center + 1.5)
    else:
        margin = max(0.5, freq_range * 0.2)
        ax1.set_ylim(max(0, freq_min - margin), freq_max + margin)

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    # ==================== 子图2: 颤音强度 ====================
    ax2 = fig.add_subplot(3, 1, 2)

    # 健康范围背景（-30 ~ -20 dB）
    ax2.fill_between([-0.5, n_files - 0.5], -30, -20,
                     color='#90EE90', alpha=0.3, label='健康范围 (-30 ~ -20 dB)')

    # 数据折线
    ax2.plot(x, vibrato_dbs, 's-', color='#FF6347', linewidth=1.5,
             markersize=3 if n_files > 50 else 5, label='调制强度', alpha=0.8)

    # 趋势线
    if n_files >= 3:
        z2 = np.polyfit(x, vibrato_dbs, 1)
        p2 = np.poly1d(z2)
        trend_dir2 = "↑" if z2[0] > 0 else "↓"
        trend_label2 = f'趋势线 ({trend_dir2} {abs(z2[0]):.4f} dB/轮)'
        ax2.plot(x, p2(x), '--', color='#8B0000', linewidth=2,
                 alpha=0.7, label=trend_label2)

    ax2.axhline(y=-20, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=-30, color='green', linestyle=':', linewidth=1.5, alpha=0.7)

    ax2.set_ylabel('调制强度 (dB)', fontsize=11)
    ax2.set_title('【颤音强度】3-8Hz调制能量占总能量的比值 | 阈值: -30~-20dB为健康',
                  fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.set_xlim(-0.5, n_files - 0.5)
    ax2.grid(True, alpha=0.3, linestyle='-')

    # 动态Y轴（放大差异）
    db_min, db_max = vibrato_dbs.min(), vibrato_dbs.max()
    db_range = db_max - db_min
    if db_range < 5:
        db_center = (db_min + db_max) / 2
        ax2.set_ylim(db_center - 8, db_center + 8)
    else:
        margin_db = max(3, db_range * 0.2)
        ax2.set_ylim(db_min - margin_db, db_max + margin_db)

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    # ==================== 子图3: 健康频率占比 ====================
    ax3 = fig.add_subplot(3, 1, 3)

    healthy_pct = healthy_ratios * 100

    # 参考区域（>50%为良好）
    ax3.axhspan(50, 100, color='#90EE90', alpha=0.2, label='良好区域 (>50%)')
    ax3.axhspan(0, 50, color='#FFB6C1', alpha=0.15, label='偏弱区域 (<50%)')

    # 数据折线
    ax3.plot(x, healthy_pct, '^-', color='#2E8B57', linewidth=1.5,
             markersize=3 if n_files > 50 else 5, label='健康频率占比', alpha=0.8)

    # 趋势线
    if n_files >= 3:
        z3 = np.polyfit(x, healthy_pct, 1)
        p3 = np.poly1d(z3)
        trend_dir3 = "↑" if z3[0] > 0 else "↓"
        trend_label3 = f'趋势线 ({trend_dir3} {abs(z3[0]):.4f} %/轮)'
        ax3.plot(x, p3(x), '--', color='#006400', linewidth=2,
                 alpha=0.7, label=trend_label3)

    ax3.axhline(y=50, color='orange', linestyle=':', linewidth=1.5, alpha=0.8)

    ax3.set_ylabel('健康频率占比 (%)', fontsize=11)
    ax3.set_xlabel('文件 (按训练轮数递增 →)', fontsize=11)
    ax3.set_title('【健康占比】4-6Hz能量在3-8Hz颤音范围中的占比 | 阈值: >50%为良好',
                  fontsize=11, fontweight='bold', pad=10)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax3.set_xlim(-0.5, n_files - 0.5)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, linestyle='-')

    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    # ==================== 说明文字（透明背景） ====================
    # 计算统计信息
    in_healthy_freq = np.sum((peak_freqs >= 4) & (peak_freqs <= 6))
    in_healthy_db = np.sum((vibrato_dbs >= -30) & (vibrato_dbs <= -20))

    desc_text = (
        "【调制频谱分析说明】\n"
        "● 原理: 分析音频的振幅包络在3-8Hz范围的调制特性，检测颤音自然度\n"
        "● 健康颤音标准: 频率4-6Hz，强度-20至-30dB\n"
        "● 判读: 频率或强度偏离健康范围表示颤音特征异常（过弱/过强/不自然）\n"
        f"● 统计: 共{n_files}个文件 | 频率达标{in_healthy_freq}个 | 强度达标{in_healthy_db}个"
    )

    fig.text(0.02, 0.005, desc_text, fontsize=9,
             verticalalignment='bottom',
             transform=fig.transFigure,
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='none',  # 透明背景
                       edgecolor='#888888',
                       linewidth=0.8))

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.11, hspace=0.35, left=0.07, right=0.98)

    # 弹出窗口显示图表
    plt.show()

    print("图表已生成并显示")
    return results


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import glob

    # 方式1: 直接指定文件列表
    # wav_files = [
    #     "path/to/epoch_100.wav",
    #     "path/to/epoch_200.wav",
    #     "path/to/epoch_300.wav",
    # ]

    # 方式2: 使用glob获取文件夹中的所有wav文件
    # wav_files = sorted(glob.glob("path/to/outputs/*.wav"))

    # 调用分析函数
    # results = analyze_modulation_spectrum(wav_files)

    print("请提供WAV文件路径数组调用 analyze_modulation_spectrum() 函数")