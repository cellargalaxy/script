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

+ 高频能量异常（HF Energy Ratio）
    + 含义：8kHz以上能量占比；NSF/GAN声码器常见问题，过高→刺耳，过低→模糊。
    + 合理分布：自然
    + 过高：齿音爆炸
    + 过低：老录音感
"""

# pip install numpy scipy matplotlib

"""
高频能量异常分析 (HF Energy Ratio)
分析AI翻唱WAV文件的8kHz以上能量占比

依赖安装:
pip install numpy scipy matplotlib

如果中文显示有问题，可额外安装:
pip install matplotlib-inline
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def _calculate_single_hf_ratio(args):
    """
    计算单个文件的高频能量比例（供进程池调用）

    返回: (文件名, 高频能量比例, 采样率, 错误信息)
    """
    wav_path, index = args
    try:
        sr, audio = wavfile.read(wav_path)

        # 立体声转单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 归一化到浮点数
        if audio.dtype == np.int16:
            audio = audio.astype(np.float64) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float64) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float64) - 128) / 128.0
        else:
            audio = audio.astype(np.float64)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))

        # 使用Welch方法计算功率谱密度（更稳定）
        nperseg = min(4096, max(256, len(audio) // 8))
        f, psd = signal.welch(audio, sr, nperseg=nperseg, noverlap=nperseg // 2)

        # 8kHz分界点
        hf_threshold = 8000
        hf_idx = np.searchsorted(f, hf_threshold)

        # 计算能量比例
        total_energy = np.trapz(psd, f)  # 使用梯形积分更准确
        hf_energy = np.trapz(psd[hf_idx:], f[hf_idx:]) if hf_idx < len(f) else 0

        hf_ratio = (hf_energy / total_energy) if total_energy > 1e-10 else 0.0

        filename = Path(wav_path).stem
        return (filename, hf_ratio, sr, None, index)

    except Exception as e:
        filename = Path(wav_path).stem if wav_path else f"file_{index}"
        return (filename, None, None, str(e), index)


def analyze_hf_energy_ratio(wav_paths: list) -> dict:
    """
    分析多个WAV文件的高频能量比例（HF Energy Ratio）

    高频能量异常 (HF Energy Ratio):
    - 含义: 8kHz以上能量占比
    - 合理范围: 5% - 15% (自然)
    - 过高 (>15%): 刺耳、齿音爆炸 (NSF/GAN声码器常见问题)
    - 过低 (<5%): 模糊、老录音感

    参数:
        wav_paths: WAV文件路径的字符串列表（已按模型轮数递增排序）

    返回:
        dict: {文件名: 高频能量比例(%)} 的字典
    """

    if not wav_paths:
        print("错误: 文件路径列表为空")
        return {}

    # ==================== 字体设置 ====================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Hiragino Sans GB',
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Arial Unicode MS',
        'DejaVu Sans', 'sans-serif'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    # ==================== 并发计算 ====================
    print(f"正在分析 {len(wav_paths)} 个文件...")

    results = {}
    args_list = [(path, i) for i, path in enumerate(wav_paths)]

    # 使用进程池并发处理
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_calculate_single_hf_ratio, args): args[1]
                   for args in args_list}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"  进度: {completed}/{len(wav_paths)}")

            filename, ratio, sr, error, idx = future.result()
            results[idx] = {
                'name': filename,
                'ratio': ratio,
                'sr': sr,
                'error': error
            }

    # 按原始顺序整理结果
    ordered_results = [results[i] for i in range(len(wav_paths))]

    # 分离有效和无效数据
    valid_data = [(r['name'], r['ratio'] * 100)
                  for r in ordered_results if r['ratio'] is not None]
    invalid_count = sum(1 for r in ordered_results if r['ratio'] is None)

    if not valid_data:
        print("错误: 没有成功分析任何文件")
        return {}

    if invalid_count > 0:
        print(f"警告: {invalid_count} 个文件分析失败")

    print(f"成功分析 {len(valid_data)}/{len(wav_paths)} 个文件")

    names, ratios = zip(*valid_data)
    names = list(names)
    ratios = np.array(ratios)

    # ==================== 图表绘制 ====================
    n_files = len(names)

    # 动态计算图表尺寸
    if n_files <= 20:
        fig_width, fig_height = 14, 9
    elif n_files <= 50:
        fig_width, fig_height = 18, 10
    else:
        fig_width, fig_height = min(24, 10 + n_files * 0.15), 11

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x = np.arange(n_files)

    # ========== 根据阈值着色 ==========
    colors = []
    for r in ratios:
        if r > 15:
            colors.append('#E74C3C')  # 红色 - 过高
        elif r < 5:
            colors.append('#F39C12')  # 橙色 - 过低
        else:
            colors.append('#27AE60')  # 绿色 - 正常

    # ========== 绘制柱状图 ==========
    bar_width = 0.7 if n_files <= 30 else 0.85
    bars = ax.bar(x, ratios, width=bar_width, color=colors,
                  alpha=0.75, edgecolor='#2C3E50', linewidth=0.5)

    # ========== 绘制趋势线 ==========
    ax.plot(x, ratios, 'o-', color='#3498DB', linewidth=1.2,
            markersize=max(2, 6 - n_files // 20), alpha=0.8, label='数值连线')

    # 移动平均线（如果数据点足够多）
    if n_files >= 7:
        window = max(3, min(7, n_files // 5))
        weights = np.ones(window) / window
        moving_avg = np.convolve(ratios, weights, mode='valid')
        ma_x = np.arange(window // 2, n_files - window // 2)
        ax.plot(ma_x, moving_avg, '-', color='#9B59B6', linewidth=2.5,
                label=f'{window}点移动平均', alpha=0.9)

    # ========== 阈值线和区域 ==========
    # 合理范围填充
    ax.axhspan(5, 15, alpha=0.12, color='#27AE60', zorder=0)

    # 阈值线
    ax.axhline(y=15, color='#E74C3C', linestyle='--', linewidth=2,
               label='过高阈值 (15%): 齿音爆炸/刺耳')
    ax.axhline(y=5, color='#F39C12', linestyle='--', linewidth=2,
               label='过低阈值 (5%): 模糊/老录音感')
    ax.axhline(y=10, color='#27AE60', linestyle=':', linewidth=1.5,
               alpha=0.7, label='理想中值 (10%)')

    # ========== Y轴范围优化（放大差异） ==========
    data_min, data_max = np.min(ratios), np.max(ratios)
    data_range = data_max - data_min

    # 差异过小时放大显示
    if data_range < 3:
        center = (data_max + data_min) / 2
        y_min = max(0, center - 4)
        y_max = center + 4
    else:
        margin = max(data_range * 0.12, 1)
        y_min = max(0, data_min - margin)
        y_max = data_max + margin

    # 确保关键阈值可见
    y_min = min(y_min, 3.5)
    y_max = max(y_max, 17)

    ax.set_ylim(y_min, y_max)

    # ========== X轴标签处理 ==========
    ax.set_xlim(-0.5, n_files - 0.5)

    if n_files <= 15:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=55, ha='right', fontsize=9)
    elif n_files <= 35:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=70, ha='right', fontsize=7)
    elif n_files <= 60:
        step = 2
        ticks = list(range(0, n_files, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([names[i] for i in ticks], rotation=75, ha='right', fontsize=6)
    else:
        step = max(2, n_files // 30)
        ticks = list(range(0, n_files, step))
        if (n_files - 1) not in ticks:
            ticks.append(n_files - 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([names[i] for i in ticks], rotation=80, ha='right', fontsize=6)

    # ========== 标题和轴标签 ==========
    ax.set_xlabel('文件 (按模型训练轮数递增 →)', fontsize=12, fontweight='medium')
    ax.set_ylabel('高频能量比例 (%)', fontsize=12, fontweight='medium')
    ax.set_title('高频能量异常分析 (HF Energy Ratio)\n'
                 '8kHz以上能量占比 · 检测NSF/GAN声码器问题',
                 fontsize=14, fontweight='bold', pad=15)

    # ========== 图例 ==========
    legend = ax.legend(loc='upper left', fontsize=9, framealpha=0.95,
                       edgecolor='#BDC3C7', fancybox=True)
    legend.get_frame().set_linewidth(0.8)

    # ========== 网格 ==========
    ax.grid(axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.3)
    ax.set_axisbelow(True)

    # ========== 统计信息 ==========
    normal_count = np.sum((ratios >= 5) & (ratios <= 15))
    high_count = np.sum(ratios > 15)
    low_count = np.sum(ratios < 5)

    stats_text = (
        f'【统计信息】\n'
        f'文件数量: {n_files}\n'
        f'平均值: {np.mean(ratios):.2f}%\n'
        f'标准差: {np.std(ratios):.2f}%\n'
        f'最小值: {data_min:.2f}% ({names[np.argmin(ratios)]})\n'
        f'最大值: {data_max:.2f}% ({names[np.argmax(ratios)]})\n'
        f'─────────────\n'
        f'正常 (5-15%): {normal_count} ({100 * normal_count / n_files:.1f}%)\n'
        f'过高 (>15%): {high_count} ({100 * high_count / n_files:.1f}%)\n'
        f'过低 (<5%): {low_count} ({100 * low_count / n_files:.1f}%)'
    )

    # ========== 指标说明（透明背景） ==========
    desc_text = (
        '【指标说明】\n'
        '高频能量比例 (HF Energy Ratio)\n'
        '计算8kHz以上频率能量占总能量的比例\n'
        '用于检测AI翻唱中声码器引入的高频异常\n\n'
        '【判断标准】\n'
        '● 绿色 (5%-15%): 正常，音质自然\n'
        '● 红色 (>15%): 过高，刺耳/齿音爆炸\n'
        '● 橙色 (<5%): 过低，模糊/老录音感\n\n'
        '【建议】\n'
        '选择指标稳定在合理范围内的模型轮次'
    )

    # 说明文字 - 透明背景
    fig.text(0.01, 0.01, desc_text, fontsize=9,
             verticalalignment='bottom', horizontalalignment='left',
             transform=fig.transFigure,
             linespacing=1.4,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='none',
                       edgecolor='none', alpha=0))

    # 统计信息框
    fig.text(0.99, 0.01, stats_text, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             transform=fig.transFigure,
             linespacing=1.3,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA',
                       edgecolor='#DEE2E6', alpha=0.95, linewidth=0.8))

    # ========== 顶部序号辅助轴 ==========
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    top_ticks = [0, n_files // 4, n_files // 2, 3 * n_files // 4, n_files - 1]
    top_ticks = sorted(set(top_ticks))
    ax_top.set_xticks(top_ticks)
    ax_top.set_xticklabels([f'#{i + 1}' for i in top_ticks], fontsize=8, color='#7F8C8D')
    ax_top.tick_params(axis='x', length=0)

    # ========== 布局调整 ==========
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, top=0.92, left=0.06, right=0.98)

    # 显示图表
    plt.show()

    # 返回结果
    return dict(zip(names, ratios.tolist()))


# ==================== 主函数入口 ====================
if __name__ == '__main__':
    import sys

    # 示例用法
    example_paths = [
        # 替换为实际的WAV文件路径
        # r"C:\path\to\model_epoch_100.wav",
        # r"C:\path\to\model_epoch_200.wav",
        # ...
    ]

    if len(sys.argv) > 1:
        # 从命令行参数获取文件路径
        wav_files = sys.argv[1:]
    elif example_paths:
        wav_files = example_paths
    else:
        print("使用方法:")
        print("  1. 直接调用函数: analyze_hf_energy_ratio(['file1.wav', 'file2.wav', ...])")
        print("  2. 命令行: python script.py file1.wav file2.wav ...")
        print("\n请提供WAV文件路径")
        sys.exit(1)

    # 运行分析
    results = analyze_hf_energy_ratio(wav_files)

    if results:
        print("\n" + "=" * 50)
        print("分析完成！各文件高频能量比例:")
        print("=" * 50)
        for name, ratio in results.items():
            status = "✓ 正常" if 5 <= ratio <= 15 else ("⚠ 过高" if ratio > 15 else "⚠ 过低")
            print(f"  {name}: {ratio:.2f}% {status}")