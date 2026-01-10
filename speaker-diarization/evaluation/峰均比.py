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

+ Crest Factor（峰均比）
    + 含义：峰值与RMS的比率，公式：Crest Factor = Peak / RMS (dB)；判断是否过度限幅或“炸声”。
    + 偏低说明“声音被糊在一起”
    + 6–10 dB：正常人声
    + 小于5 dB：过度压缩
    + 大于12 dB：动态失控
"""

# pip install numpy scipy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


def analyze_crest_factor(wav_paths: List[str]) -> Dict:
    """
    分析多个WAV文件的Crest Factor（峰均比）并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数递增排序）

    返回:
        包含分析结果的字典
    """

    # ==================== 内部函数定义 ====================

    def calculate_single_file(args: Tuple[int, str]) -> Dict:
        """计算单个WAV文件的Crest Factor"""
        idx, wav_path = args
        try:
            sample_rate, data = wavfile.read(wav_path)

            # 立体声转单声道
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # 归一化到 [-1, 1]
            if data.dtype == np.int16:
                data = data.astype(np.float64) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float64) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float64) - 128) / 128.0
            elif data.dtype in [np.float32, np.float64]:
                data = data.astype(np.float64)
            else:
                data = data.astype(np.float64) / np.max(np.abs(data))

            # 计算峰值和RMS
            peak = np.max(np.abs(data))
            rms = np.sqrt(np.mean(data ** 2))

            # 计算Crest Factor (dB)
            if rms > 1e-10 and peak > 1e-10:
                crest_factor_db = 20 * np.log10(peak / rms)
            else:
                crest_factor_db = 0.0

            return {
                'index': idx,
                'path': wav_path,
                'filename': os.path.basename(wav_path),
                'crest_factor': crest_factor_db,
                'peak': peak,
                'rms': rms,
                'sample_rate': sample_rate,
                'duration': len(data) / sample_rate,
                'error': None
            }
        except Exception as e:
            return {
                'index': idx,
                'path': wav_path,
                'filename': os.path.basename(wav_path),
                'crest_factor': None,
                'error': str(e)
            }

    def get_quality_color(cf: float) -> str:
        """根据Crest Factor值返回对应的颜色"""
        if cf < 5:
            return '#E74C3C'  # 红色 - 过度压缩
        elif 6 <= cf <= 10:
            return '#27AE60'  # 绿色 - 正常
        elif 10 < cf <= 12:
            return '#F39C12'  # 橙色 - 偏高
        else:
            return '#9B59B6'  # 紫色 - 动态失控

    def get_quality_label(cf: float) -> str:
        """根据Crest Factor值返回质量标签"""
        if cf < 5:
            return '过度压缩'
        elif cf < 6:
            return '略低'
        elif cf <= 10:
            return '正常'
        elif cf <= 12:
            return '偏高'
        else:
            return '动态失控'

    # ==================== 并发计算 ====================

    print(f"正在分析 {len(wav_paths)} 个音频文件...")

    # 使用线程池并发处理
    max_workers = min(os.cpu_count() or 4, 16, len(wav_paths))
    results_list = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(calculate_single_file, (i, path)): i
                   for i, path in enumerate(wav_paths)}

        completed = 0
        for future in as_completed(futures):
            results_list.append(future.result())
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"  进度: {completed}/{len(wav_paths)}")

    # 按原始顺序排序
    results_list.sort(key=lambda x: x['index'])

    # 分离有效和无效结果
    valid_results = [r for r in results_list if r['error'] is None]
    error_results = [r for r in results_list if r['error'] is not None]

    if not valid_results:
        print("错误: 没有有效的WAV文件可供分析")
        return {'valid': [], 'errors': error_results}

    # ==================== 数据准备 ====================

    n_files = len(valid_results)
    filenames = [r['filename'] for r in valid_results]
    crest_factors = np.array([r['crest_factor'] for r in valid_results])
    colors = [get_quality_color(cf) for cf in crest_factors]

    # 统计信息
    cf_mean = np.mean(crest_factors)
    cf_std = np.std(crest_factors)
    cf_min = np.min(crest_factors)
    cf_max = np.max(crest_factors)

    # 区间统计
    count_over_compressed = np.sum(crest_factors < 5)
    count_low = np.sum((crest_factors >= 5) & (crest_factors < 6))
    count_normal = np.sum((crest_factors >= 6) & (crest_factors <= 10))
    count_high = np.sum((crest_factors > 10) & (crest_factors <= 12))
    count_out_of_control = np.sum(crest_factors > 12)

    # ==================== 可视化 ====================

    # 设置中文字体（尝试多种字体）
    font_candidates = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC',
        'Hiragino Sans GB', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans'
    ]

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_candidates
    plt.rcParams['axes.unicode_minus'] = False

    # 动态计算图表尺寸
    fig_width = max(16, min(28, n_files * 0.25 + 6))
    fig_height = 11

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')

    x = np.arange(n_files)

    # ---- 绘制参考区域和阈值线 ----

    # 正常范围填充
    ax.axhspan(6, 10, alpha=0.12, color='#27AE60', zorder=1, label='正常人声范围 (6-10 dB)')

    # 阈值线
    threshold_lines = [
        (5, '#E74C3C', '过度压缩阈值'),
        (6, '#27AE60', '正常下限'),
        (10, '#27AE60', '正常上限'),
        (12, '#9B59B6', '动态失控阈值'),
    ]

    for y_val, color, label in threshold_lines:
        ax.axhline(y=y_val, color=color, linestyle='--', linewidth=1.8, alpha=0.7, zorder=2)

    # ---- 绘制数据 ----

    # 趋势线
    ax.plot(x, crest_factors, color='#3498DB', linewidth=1.5, alpha=0.5, zorder=3)

    # 数据点
    scatter = ax.scatter(x, crest_factors, c=colors, s=70,
                         edgecolors='white', linewidths=1, zorder=4)

    # 标注最大最小值
    idx_min = np.argmin(crest_factors)
    idx_max = np.argmax(crest_factors)

    ax.annotate(f'最小: {cf_min:.1f}dB\n{filenames[idx_min]}',
                xy=(idx_min, cf_min), xytext=(idx_min, cf_min - 1.5),
                fontsize=8, ha='center', color='#E74C3C',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#E74C3C', alpha=0.9))

    ax.annotate(f'最大: {cf_max:.1f}dB\n{filenames[idx_max]}',
                xy=(idx_max, cf_max), xytext=(idx_max, cf_max + 1.5),
                fontsize=8, ha='center', color='#9B59B6',
                arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#9B59B6', alpha=0.9))

    # ---- 右侧阈值标注 ----

    ax.text(n_files + 0.8, 5, '5 dB 过度压缩', va='center', fontsize=9,
            color='#E74C3C', fontweight='bold')
    ax.text(n_files + 0.8, 6, '6 dB 正常下限', va='center', fontsize=9,
            color='#27AE60', fontweight='bold')
    ax.text(n_files + 0.8, 10, '10 dB 正常上限', va='center', fontsize=9,
            color='#27AE60', fontweight='bold')
    ax.text(n_files + 0.8, 12, '12 dB 动态失控', va='center', fontsize=9,
            color='#9B59B6', fontweight='bold')

    # ---- 坐标轴设置 ----

    ax.set_xlabel('文件序号（按模型训练轮数递增 →）', fontsize=12, fontweight='bold')
    ax.set_ylabel('Crest Factor (dB)', fontsize=12, fontweight='bold')
    ax.set_title('AI翻唱音频 Crest Factor（峰均比）质量评估',
                 fontsize=16, fontweight='bold', pad=20)

    # X轴刻度标签
    if n_files <= 20:
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=55, ha='right', fontsize=8)
    elif n_files <= 50:
        step = 2
        ax.set_xticks(x[::step])
        labels = [filenames[i] for i in range(0, n_files, step)]
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=7)
    else:
        step = max(2, n_files // 25)
        ax.set_xticks(x[::step])
        labels = [filenames[i] for i in range(0, n_files, step)]
        ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=7)

    # Y轴范围（动态调整以突出差异）
    y_data_range = cf_max - cf_min
    if y_data_range < 2:
        y_padding = 2
    else:
        y_padding = y_data_range * 0.25

    y_lower = max(0, min(cf_min - y_padding, 4))
    y_upper = max(cf_max + y_padding, 13)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlim(-1, n_files + 5)

    # 网格
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    # ---- 说明文字框（背景透明）----

    description = (
        "【Crest Factor（峰均比）指标说明】\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "计算公式: CF = 20 × log₁₀(Peak ÷ RMS) dB\n"
        "检测目的: 判断是否过度限幅或出现炸声\n"
    "\n"
    "【阈值参考】\n"
    "  ● 6 - 10 dB : 正常人声范围\n"
    "  ● < 5 dB    : 过度压缩（声音糊在一起）\n"
    "  ● > 12 dB   : 动态失控\n"
    "\n"
    "【颜色说明】\n"
    "  🟢 绿色: 正常 (6-10 dB)\n"
    "  🟠 橙色: 偏高 (10-12 dB)\n"
    "  🔴 红色: 过度压缩 (<5 dB)\n"
    "  🟣 紫色: 动态失控 (>12 dB)"
    )

    ax.text(0.01, 0.99, description, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='none',
                      edgecolor='#BDC3C7', linewidth=1.5))

    # ---- 统计信息框 ----

    stats_text = (
        f"【统计信息】\n"
        f"文件总数: {n_files}\n"
        f"均值: {cf_mean:.2f} dB\n"
        f"标准差: {cf_std:.2f} dB\n"
        f"范围: {cf_min:.2f} - {cf_max:.2f} dB\n"
        f"\n"
        f"【分布统计】\n"
        f"过度压缩 (<5dB): {count_over_compressed} ({count_over_compressed / n_files * 100:.0f}%)\n"
        f"正常范围 (6-10dB): {count_normal} ({count_normal / n_files * 100:.0f}%)\n"
        f"偏高 (10-12dB): {count_high} ({count_high / n_files * 100:.0f}%)\n"
        f"动态失控 (>12dB): {count_out_of_control} ({count_out_of_control / n_files * 100:.0f}%)"
    )

    ax.text(0.99, 0.99, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='none',
                      edgecolor='#BDC3C7', linewidth=1.5))

    # ---- 布局调整 ----

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, right=0.94, top=0.92, left=0.06)

    # 显示图表
    plt.show()

    # ==================== 打印报告 ====================

    print("\n" + "=" * 70)
    print("                  Crest Factor（峰均比）分析报告")
    print("=" * 70)
    print(f"\n📊 基本统计")
    print(f"   有效文件数: {n_files} / {len(wav_paths)}")
    print(f"   数值范围:   {cf_min:.2f} ~ {cf_max:.2f} dB")
    print(f"   平均值:     {cf_mean:.2f} dB")
    print(f"   标准差:     {cf_std:.2f} dB")

    print(f"\n📈 质量分布")
    print(f"   🔴 过度压缩 (<5 dB):    {count_over_compressed:3d} 个 ({count_over_compressed / n_files * 100:5.1f}%)")
    print(f"   🟡 略低 (5-6 dB):       {count_low:3d} 个 ({count_low / n_files * 100:5.1f}%)")
    print(f"   🟢 正常 (6-10 dB):      {count_normal:3d} 个 ({count_normal / n_files * 100:5.1f}%)")
    print(f"   🟠 偏高 (10-12 dB):     {count_high:3d} 个 ({count_high / n_files * 100:5.1f}%)")
    print(f"   🟣 动态失控 (>12 dB):   {count_out_of_control:3d} 个 ({count_out_of_control / n_files * 100:5.1f}%)")

    # 找出最佳文件（最接近8dB，正常范围中心）
    ideal_cf = 8.0
    best_idx = np.argmin(np.abs(crest_factors - ideal_cf))
    print(f"\n🏆 最佳文件（最接近理想值 8 dB）:")
    print(f"   {valid_results[best_idx]['filename']}")
    print(f"   Crest Factor: {crest_factors[best_idx]:.2f} dB")

    if error_results:
        print(f"\n⚠️ 处理失败的文件: {len(error_results)} 个")
        for r in error_results[:5]:
            print(f"   - {r['filename']}: {r['error']}")
        if len(error_results) > 5:
            print(f"   ... 还有 {len(error_results) - 5} 个文件")

    print("\n" + "=" * 70)

    # 返回结果
    return {
        'valid_results': valid_results,
        'error_results': error_results,
        'statistics': {
            'mean': cf_mean,
            'std': cf_std,
            'min': cf_min,
            'max': cf_max,
            'best_file': valid_results[best_idx]['filename'],
            'best_value': crest_factors[best_idx]
        },
        'distribution': {
            'over_compressed': count_over_compressed,
            'low': count_low,
            'normal': count_normal,
            'high': count_high,
            'out_of_control': count_out_of_control
        }
    }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例：使用方法
    import glob

    # 方式1: 直接提供文件路径列表
    wav_files = [
        r"C:\audio\model_epoch_100.wav",
        r"C:\audio\model_epoch_200.wav",
        r"C:\audio\model_epoch_300.wav",
        # ... 更多文件
    ]

    # 方式2: 使用glob匹配文件
    # wav_files = sorted(glob.glob(r"C:\audio\*.wav"))

    # 执行分析
    # results = analyze_crest_factor(wav_files)

    print("请将 wav_files 替换为实际的文件路径列表，然后运行 analyze_crest_factor(wav_files)")