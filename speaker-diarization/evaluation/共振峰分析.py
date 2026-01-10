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

+ 共振峰分析/连续性（Formant Analysis）
    + 含义：声道共振峰（F1-F3）轨迹，使用LPC提取；检测音色辨识和转换自然度，跳变表示不连贯。
    + 共振峰 男正常范围 女正常范围 异常判断
    + F1 300–800Hz 350–900Hz 开口度异常
    + F2 800–2500Hz 900–2800Hz 舌位异常
    + F3 2000–3500Hz 2500–3800Hz 音色缺失/规整呆板
"""

# pip install numpy matplotlib praat-parselmouth

"""
共振峰分析 (Formant Analysis) - AI翻唱质量评估

依赖安装:
    pip install numpy matplotlib praat-parselmouth

使用方法:
    from formant_analysis import analyze_formant_quality
    results = analyze_formant_quality(['path1.wav', 'path2.wav', ...])
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


def analyze_formant_quality(wav_paths):
    """
    分析多个WAV文件的共振峰质量并可视化对比

    共振峰分析说明:
        - 使用LPC提取声道共振峰（F1-F3）轨迹
        - 检测音色辨识和转换自然度
        - 跳变表示不连贯

    共振峰正常范围:
        F1: 男300-800Hz, 女350-900Hz → 开口度异常判断
        F2: 男800-2500Hz, 女900-2800Hz → 舌位异常判断
        F3: 男2000-3500Hz, 女2500-3800Hz → 音色缺失/规整呆板判断

    参数:
        wav_paths (list): wav文件路径的字符串数组（应按模型轮数排序）

    返回:
        list: 包含各文件共振峰指标的字典列表
    """

    # ==================== 依赖检查 ====================
    try:
        import parselmouth
        from parselmouth.praat import call as praat_call
    except ImportError:
        raise ImportError(
            "缺少依赖库 praat-parselmouth\n"
            "请运行: pip install praat-parselmouth"
        )

    # ==================== 字体设置 ====================
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'PingFang SC',
        'Hiragino Sans GB', 'DejaVu Sans', 'Arial'
    ]
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    # ==================== 常量定义 ====================
    # 共振峰正常范围（综合男女）
    FORMANT_RANGES = {
        'F1': (300, 900),  # 开口度
        'F2': (800, 2800),  # 舌位
        'F3': (2000, 3800)  # 音色
    }
    JUMP_THRESHOLD = 100  # Hz，跳变判断阈值

    # ==================== 内部函数 ====================

    def extract_formants_single(wav_path):
        """提取单个文件的共振峰"""
        import parselmouth
        from parselmouth.praat import call as pc

        try:
            sound = parselmouth.Sound(wav_path)
            # Burg方法: 时间步长自动, 最大5个共振峰, 最大频率5500Hz, 窗口25ms, 预加重50Hz
            formant = pc(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)

            num_frames = pc(formant, "Get number of frames")
            f1, f2, f3 = [], [], []

            for i in range(1, num_frames + 1):
                t = pc(formant, "Get time from frame number", i)
                f1.append(pc(formant, "Get value at time", 1, t, "Hertz", "Linear"))
                f2.append(pc(formant, "Get value at time", 2, t, "Hertz", "Linear"))
                f3.append(pc(formant, "Get value at time", 3, t, "Hertz", "Linear"))

            return wav_path, np.array(f1), np.array(f2), np.array(f3), None
        except Exception as e:
            return wav_path, None, None, None, str(e)

    def compute_metrics(f1, f2, f3, filename):
        """计算共振峰质量指标"""

        def safe_array(arr):
            if arr is None:
                return np.array([])
            return arr[~np.isnan(arr)]

        f1_v, f2_v, f3_v = safe_array(f1), safe_array(f2), safe_array(f3)
        m = {'filename': filename}

        for name, arr, (lo, hi) in [
            ('F1', f1_v, FORMANT_RANGES['F1']),
            ('F2', f2_v, FORMANT_RANGES['F2']),
            ('F3', f3_v, FORMANT_RANGES['F3'])
        ]:
            n = len(arr)
            # 均值
            m[f'{name}_mean'] = float(np.mean(arr)) if n > 0 else np.nan
            # 标准差
            m[f'{name}_std'] = float(np.std(arr)) if n > 0 else np.nan
            # 帧间差异均值（连续性，越小越好）
            m[f'{name}_continuity'] = float(np.mean(np.abs(np.diff(arr)))) if n > 1 else np.nan
            # 跳变率（越低越好）
            m[f'{name}_jump_rate'] = float(np.mean(np.abs(np.diff(arr)) > JUMP_THRESHOLD)) if n > 1 else 0.0
            # 正常范围比例（越高越好）
            m[f'{name}_normal_ratio'] = float(np.mean((arr >= lo) & (arr <= hi))) if n > 0 else np.nan

        return m

    def create_visualization(results):
        """创建可视化图表"""
        n = len(results)
        x = np.arange(n)

        def get_metric(key):
            return np.array([r.get(key, np.nan) for r in results])

        # 简化文件名
        filenames = [r.get('filename', f'{i}') for i, r in enumerate(results)]
        short_names = []
        for name in filenames:
            base = os.path.splitext(name)[0]
            short_names.append(base[:16] + '..' if len(base) > 18 else base)

        # 创建 4x3 子图
        fig, axes = plt.subplots(4, 3, figsize=(24, 22))
        fig.suptitle(
            '共振峰分析 (Formant Analysis) - AI翻唱质量评估\n'
            '█ 声道共振峰F1-F3轨迹分析 | 检测音色辨识和转换自然度 | 跳变表示不连贯',
            fontsize=15, fontweight='bold', y=0.995
        )

        # 描述文字
        desc = {
            'F1_mean': '【开口度指标】\n正常范围: 300-900Hz\n• 男: 300-800Hz\n• 女: 350-900Hz\n↓低于范围: 开口度不足\n↑高于范围: 开口度过大',
            'F2_mean': '【舌位指标】\n正常范围: 800-2800Hz\n• 男: 800-2500Hz\n• 女: 900-2800Hz\n反映舌头前后位置控制',
            'F3_mean': '【音色指标】\n正常范围: 2000-3800Hz\n• 男: 2000-3500Hz\n• 女: 2500-3800Hz\n↓过低: 音色缺失\n过于稳定: 规整呆板',
            'F1_cont': '【F1连续性】\n帧间差异均值(Hz)\n• 越小越平滑自然\n• 理想值 <30Hz\n• >50Hz 可能有跳变',
            'F2_cont': '【F2连续性】\n帧间差异均值(Hz)\n• 越小越平滑自然\n• 理想值 <50Hz',
            'F3_cont': '【F3连续性】\n帧间差异均值(Hz)\n• 越小越平滑自然\n• 理想值 <60Hz',
            'F1_jump': '【F1跳变率】\n跳变阈值: 100Hz\n• 越低越连贯\n• >0.3 严重不连贯\n• 理想值 <0.1',
            'F2_jump': '【F2跳变率】\n跳变阈值: 100Hz\n• 越低越连贯\n• 理想值 <0.15',
            'F3_jump': '【F3跳变率】\n跳变阈值: 100Hz\n• 越低越连贯\n• 理想值 <0.2',
            'F1_norm': '【F1范围符合度】\n在正常范围内的帧比例\n• 越高越好\n• 1.0 = 完全正常\n• <0.7 需关注',
            'F2_norm': '【F2范围符合度】\n在正常范围内的帧比例\n• 越高越好\n• 1.0 = 完全正常',
            'F3_norm': '【F3范围符合度】\n在正常范围内的帧比例\n• 越高越好\n• 1.0 = 完全正常',
        }

        def set_xlabels(ax):
            """设置X轴标签"""
            if n <= 15:
                ax.set_xticks(x)
                ax.set_xticklabels(short_names, rotation=55, ha='right', fontsize=6)
            else:
                step = max(n // 12, 1)
                ticks = list(range(0, n, step))
                if n - 1 not in ticks:
                    ticks.append(n - 1)
                ax.set_xticks(ticks)
                ax.set_xticklabels([short_names[i] for i in ticks], rotation=55, ha='right', fontsize=6)

        def add_desc_box(ax, text):
            """添加描述文字框（透明背景）"""
            ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', fontfamily='sans-serif',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                              edgecolor='gray', alpha=0.0))

        def plot_mean_with_range(ax, data, title, formant_key, desc_key):
            """绘制带正常范围的均值图"""
            valid = ~np.isnan(data)
            v = data[valid]

            if len(v) == 0:
                ax.text(0.5, 0.5, '无有效数据', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                return

            lo, hi = FORMANT_RANGES[formant_key]

            # 计算数据范围和正常范围的关系
            data_min, data_max = v.min(), v.max()
            data_range = data_max - data_min

            # 计算显示范围：主要根据数据范围来设定，同时考虑正常范围作为参考
            display_min = min(data_min, lo) - data_range * 0.15
            display_max = max(data_max, hi) + data_range * 0.15

            # 确保显示范围有最小宽度，避免数据过于接近时看不出来
            min_display_range = max(data_range * 1.5, 50)  # 至少50Hz的显示范围
            current_display_range = display_max - display_min
            if current_display_range < min_display_range:
                # 扩展显示范围到最小要求
                center = (display_max + display_min) / 2
                display_min = center - min_display_range / 2
                display_max = center + min_display_range / 2

            # 绘制正常范围带（仅在数据范围与正常范围有重叠时才显示）
            if hi > display_min and lo < display_max:
                visible_lo = max(lo, display_min)
                visible_hi = min(hi, display_max)
                if visible_hi > visible_lo:
                    ax.axhspan(visible_lo, visible_hi, alpha=0.15, color='green', label='正常范围')
                    # 仅在正常范围在可见区域内时才绘制边界线
                    if lo > display_min:
                        ax.axhline(lo, color='green', ls='--', alpha=0.3, lw=1)
                    if hi < display_max:
                        ax.axhline(hi, color='green', ls='--', alpha=0.3, lw=1)

            # 数据线
            ax.plot(x[valid], data[valid], 'b-', lw=1.5, marker='o', ms=4, alpha=0.85, label='实测值')

            # 趋势线
            if np.sum(valid) >= 5:
                z = np.polyfit(x[valid], data[valid], 1)
                p = np.poly1d(z)
                trend_line = ax.plot(x, p(x), 'r--', lw=2, alpha=0.7, label='趋势线')

            # 设置Y轴范围
            ax.set_ylim(display_min, display_max)

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('模型轮数 (按训练进度排序)', fontsize=9)
            ax.set_ylabel('频率 (Hz)', fontsize=9)
            ax.grid(True, alpha=0.35, linestyle='-', lw=0.5)
            ax.legend(loc='lower right', fontsize=7, framealpha=0.8)
            add_desc_box(ax, desc[desc_key])
            set_xlabels(ax)

        def plot_continuity(ax, data, title, desc_key):
            """绘制连续性图"""
            valid = ~np.isnan(data)
            v = data[valid]

            if len(v) == 0:
                ax.text(0.5, 0.5, '无有效数据', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                return

            ax.plot(x[valid], data[valid], 'b-', lw=1.5, marker='o', ms=4, alpha=0.85)

            # 趋势线
            if np.sum(valid) >= 5:
                z = np.polyfit(x[valid], data[valid], 1)
                p = np.poly1d(z)
                improving = z[0] < 0
                color = 'green' if improving else 'red'
                label = '趋势↓改善' if improving else '趋势↑恶化'
                ax.plot(x, p(x), '--', color=color, lw=2, alpha=0.7, label=label)

            # Y轴范围（动态适配数据）
            if len(v) > 0:
                rng = v.max() - v.min()
                # 如果数据范围太小，扩大显示范围以突出差异
                if rng < v.mean() * 0.1:
                    center = (v.max() + v.min()) / 2
                    display_range = max(rng * 3, center * 0.2)
                    display_min = max(0, center - display_range/2)
                    display_max = center + display_range/2
                else:
                    margin = rng * 0.25
                    display_min = max(0, v.min() - margin)
                    display_max = v.max() + margin
                ax.set_ylim(display_min, display_max)

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('模型轮数 (按训练进度排序)', fontsize=9)
            ax.set_ylabel('帧间差异 (Hz)', fontsize=9)
            ax.grid(True, alpha=0.35, linestyle='-', lw=0.5)
            if np.sum(valid) >= 5:
                ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
            add_desc_box(ax, desc[desc_key])
            set_xlabels(ax)

        def plot_rate(ax, data, title, desc_key, lower_better=True):
            """绘制比率图"""
            valid = ~np.isnan(data)
            v = data[valid]

            if len(v) == 0:
                ax.text(0.5, 0.5, '无有效数据', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                return

            ax.plot(x[valid], data[valid], 'b-', lw=1.5, marker='o', ms=4, alpha=0.85)

            # 趋势线
            if np.sum(valid) >= 5:
                z = np.polyfit(x[valid], data[valid], 1)
                p = np.poly1d(z)
                improving = (lower_better and z[0] < 0) or (not lower_better and z[0] > 0)
                color = 'green' if improving else 'red'
                label = '趋势改善 ✓' if improving else '趋势恶化 ✗'
                ax.plot(x, p(x), '--', color=color, lw=2, alpha=0.7, label=label)

            # Y轴范围（动态适配数据）
            if len(v) > 0:
                v_range = v.max() - v.min()
                # 如果数据差异太小，扩大显示范围以突出差异
                if v_range < 0.05:
                    center = (v.max() + v.min()) / 2
                    display_range = max(v_range * 3, 0.1)
                    display_min = max(0, center - display_range/2)
                    display_max = min(1, center + display_range/2)
                else:
                    margin = v_range * 0.2
                    display_min = max(0, v.min() - margin)
                    display_max = min(1, v.max() + margin)

                # 确保显示范围合理
                if display_max - display_min < 0.1:
                    center = (display_min + display_max) / 2
                    display_min = max(0, center - 0.15)
                    display_max = min(1, center + 0.15)

                ax.set_ylim(display_min, display_max)
            else:
                ax.set_ylim(0, 1)

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('模型轮数 (按训练进度排序)', fontsize=9)
            ax.set_ylabel('比率 (0-1)', fontsize=9)
            ax.grid(True, alpha=0.35, linestyle='-', lw=0.5)
            if np.sum(valid) >= 5:
                loc = 'upper right' if lower_better else 'lower right'
                ax.legend(loc=loc, fontsize=7, framealpha=0.8)
            add_desc_box(ax, desc[desc_key])
            set_xlabels(ax)

        # ===== 绘制图表 =====
        # 第1行: 共振峰均值
        plot_mean_with_range(axes[0, 0], get_metric('F1_mean'), 'F1 均值 - 开口度', 'F1', 'F1_mean')
        plot_mean_with_range(axes[0, 1], get_metric('F2_mean'), 'F2 均值 - 舌位', 'F2', 'F2_mean')
        plot_mean_with_range(axes[0, 2], get_metric('F3_mean'), 'F3 均值 - 音色', 'F3', 'F3_mean')

        # 第2行: 连续性（帧间差异）
        plot_continuity(axes[1, 0], get_metric('F1_continuity'), 'F1 帧间差异 - 连续性', 'F1_cont')
        plot_continuity(axes[1, 1], get_metric('F2_continuity'), 'F2 帧间差异 - 连续性', 'F2_cont')
        plot_continuity(axes[1, 2], get_metric('F3_continuity'), 'F3 帧间差异 - 连续性', 'F3_cont')

        # 第3行: 跳变率
        plot_rate(axes[2, 0], get_metric('F1_jump_rate'), 'F1 跳变率 - 不连贯度', 'F1_jump', lower_better=True)
        plot_rate(axes[2, 1], get_metric('F2_jump_rate'), 'F2 跳变率 - 不连贯度', 'F2_jump', lower_better=True)
        plot_rate(axes[2, 2], get_metric('F3_jump_rate'), 'F3 跳变率 - 不连贯度', 'F3_jump', lower_better=True)

        # 第4行: 正常范围比例
        plot_rate(axes[3, 0], get_metric('F1_normal_ratio'), 'F1 正常范围比例', 'F1_norm', lower_better=False)
        plot_rate(axes[3, 1], get_metric('F2_normal_ratio'), 'F2 正常范围比例', 'F2_norm', lower_better=False)
        plot_rate(axes[3, 2], get_metric('F3_normal_ratio'), 'F3 正常范围比例', 'F3_norm', lower_better=False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.94, hspace=0.42, wspace=0.22)
        plt.show()

    # ==================== 主处理流程 ====================

    print("=" * 60)
    print("  共振峰分析 (Formant Analysis) - AI翻唱质量评估")
    print("=" * 60)
    print(f"待处理文件数: {len(wav_paths)}")
    print(f"并发线程数: {min(os.cpu_count() or 4, 8)}")
    print("-" * 60)

    # 并发提取共振峰
    formant_data = {}
    max_workers = min(os.cpu_count() or 4, 8)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_formants_single, p): p for p in wav_paths}
        completed = 0
        for future in as_completed(futures):
            path, f1, f2, f3, error = future.result()
            formant_data[path] = (f1, f2, f3, error)
            completed += 1
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"  提取进度: {completed}/{len(wav_paths)} ({100 * completed // len(wav_paths)}%)")

    # 计算指标
    results = []
    errors = []
    for path in wav_paths:
        f1, f2, f3, error = formant_data[path]
        filename = os.path.basename(path)
        if error:
            errors.append((filename, error))
            results.append({'filename': filename, 'error': error})
        else:
            metrics = compute_metrics(f1, f2, f3, filename)
            results.append(metrics)

    print("-" * 60)
    if errors:
        print(f"⚠ 警告: {len(errors)} 个文件处理失败:")
        for fname, err in errors[:3]:
            print(f"   • {fname}: {err}")
        if len(errors) > 3:
            print(f"   ... 及其他 {len(errors) - 3} 个文件")

    print(f"✓ 成功处理: {len(wav_paths) - len(errors)} 个文件")
    print("-" * 60)
    print("正在生成可视化图表...")

    # 可视化
    create_visualization(results)

    print("=" * 60)
    print("  共振峰分析完成!")
    print("=" * 60)

    return results


# ==================== 直接运行入口 ====================
if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) > 1:
        # 支持通配符
        paths = []
        for arg in sys.argv[1:]:
            paths.extend(glob.glob(arg))
        paths = sorted(set(paths))

        if paths:
            analyze_formant_quality(paths)
        else:
            print("未找到匹配的文件")
    else:
        print("=" * 50)
        print("共振峰分析 (Formant Analysis)")
        print("=" * 50)
        print("\n用法:")
        print("  python formant_analysis.py file1.wav file2.wav ...")
        print("  python formant_analysis.py ./output/*.wav")
        print("\n或在代码中导入:")
        print("  from formant_analysis import analyze_formant_quality")
        print("  results = analyze_formant_quality(wav_paths)")