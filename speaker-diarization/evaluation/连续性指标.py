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

+ 连续性指标（Frame-level Continuity）
    + 含义：帧间余弦相似度，公式：`Continuity = [向量(t) · 向量(t+1)] / [||t|| × ||t+1||]`；检测断裂。
    + 大于0.95：非常连贯
    + 0.90–0.95：正常
    + 0.85–0.90：轻微断裂
    + 小于0.85：连贯性差
"""

# pip install numpy librosa matplotlib

"""
帧间连续性指标分析工具 (Frame-level Continuity Analyzer)

依赖安装：
pip install numpy librosa matplotlib

"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


def analyze_frame_continuity(wav_paths: List[str]) -> Optional[List[Dict]]:
    """
    分析多个WAV文件的帧间连续性指标（Frame-level Continuity）

    连续性指标说明：
    - 计算公式：Continuity = [向量(t) · 向量(t+1)] / [||t|| × ||t+1||]
    - 用于检测音频的断裂情况
    - 大于0.95：非常连贯
    - 0.90–0.95：正常
    - 0.85–0.90：轻微断裂
    - 小于0.85：连贯性差

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数递增排序）

    返回:
        分析结果列表，每个元素包含文件的各项指标
    """

    # ==================== 内部函数定义 ====================

    def compute_single_file(args) -> Dict:
        """计算单个文件的连续性指标"""
        idx, wav_path = args
        try:
            # 加载音频
            y, sr = librosa.load(wav_path, sr=None)

            # 提取MFCC特征（13维，每帧一个向量）
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # 计算相邻帧的余弦相似度
            n_frames = mfcc.shape[1]
            if n_frames < 2:
                raise ValueError("音频帧数不足")

            similarities = np.zeros(n_frames - 1)

            for t in range(n_frames - 1):
                vec_t = mfcc[:, t]
                vec_t1 = mfcc[:, t + 1]
                norm_t = np.linalg.norm(vec_t)
                norm_t1 = np.linalg.norm(vec_t1)

                if norm_t > 1e-10 and norm_t1 > 1e-10:
                    similarities[t] = np.dot(vec_t, vec_t1) / (norm_t * norm_t1)
                else:
                    similarities[t] = 0.0

            return {
                'index': idx,
                'path': wav_path,
                'filename': Path(wav_path).stem,
                'mean': float(np.mean(similarities)),
                'min': float(np.min(similarities)),
                'max': float(np.max(similarities)),
                'median': float(np.median(similarities)),
                'std': float(np.std(similarities)),
                'percentile_5': float(np.percentile(similarities, 5)),
                'percentile_10': float(np.percentile(similarities, 10)),
                'below_085_pct': float(np.mean(similarities < 0.85) * 100),
                'below_090_pct': float(np.mean(similarities < 0.90) * 100),
                'below_095_pct': float(np.mean(similarities < 0.95) * 100),
                'n_frames': n_frames,
                'duration': float(len(y) / sr),
                'success': True
            }
        except Exception as e:
            return {
                'index': idx,
                'path': wav_path,
                'filename': Path(wav_path).stem,
                'error': str(e),
                'success': False
            }

    def setup_matplotlib():
        """配置matplotlib使用常规字体"""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'PingFang SC',
            'Hiragino Sans GB', 'WenQuanYi Micro Hei',
            'Noto Sans CJK SC', 'DejaVu Sans', 'Arial'
        ]
        plt.rcParams['axes.unicode_minus'] = False

    def get_quality_color(value: float) -> str:
        """根据连续性值返回对应颜色"""
        if value >= 0.95:
            return '#27ae60'  # 绿色 - 非常连贯
        elif value >= 0.90:
            return '#2ecc71'  # 浅绿 - 正常
        elif value >= 0.85:
            return '#f39c12'  # 橙色 - 轻微断裂
        else:
            return '#e74c3c'  # 红色 - 连贯性差

    def create_visualization(valid_results: List[Dict]):
        """创建可视化图表"""
        setup_matplotlib()

        n_files = len(valid_results)

        # 提取数据
        filenames = [r['filename'] for r in valid_results]
        means = np.array([r['mean'] for r in valid_results])
        mins = np.array([r['min'] for r in valid_results])
        stds = np.array([r['std'] for r in valid_results])
        percentile_5 = np.array([r['percentile_5'] for r in valid_results])
        below_085 = np.array([r['below_085_pct'] for r in valid_results])
        below_090 = np.array([r['below_090_pct'] for r in valid_results])
        below_095 = np.array([r['below_095_pct'] for r in valid_results])

        # 计算综合评分 (0-100)
        scores = np.clip((means - 0.80) / 0.20 * 60 +
                         (1 - below_085 / 100) * 20 +
                         (1 - below_090 / 100) * 20, 0, 100)

        # 计算图表尺寸
        fig_width = 20
        bar_chart_height = max(8, n_files * 0.18)
        fig_height = 14 + bar_chart_height * 0.5

        fig = plt.figure(figsize=(fig_width, fig_height))

        # 使用GridSpec布局
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, max(1.2, n_files * 0.02)],
                              hspace=0.35, wspace=0.25,
                              left=0.06, right=0.98, top=0.93, bottom=0.03)

        x = np.arange(n_files)
        marker_size = max(2, 8 - n_files // 15)

        # ==================== 图1: 平均连续性趋势 ====================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(x, means, 'b-o', markersize=marker_size,
                 linewidth=1.5, label='平均连续性', zorder=3)
        ax1.fill_between(x, means - stds, np.minimum(means + stds, 1.0),
                         alpha=0.25, color='blue', label='±1σ 范围')

        # 阈值线
        ax1.axhline(y=0.95, color='#27ae60', linestyle='--', linewidth=2, label='0.95 非常连贯')
        ax1.axhline(y=0.90, color='#f39c12', linestyle='--', linewidth=2, label='0.90 正常')
        ax1.axhline(y=0.85, color='#e74c3c', linestyle='--', linewidth=2, label='0.85 轻微断裂')

        # 背景色块
        ax1.axhspan(0.95, 1.0, alpha=0.1, color='#27ae60')
        ax1.axhspan(0.90, 0.95, alpha=0.1, color='#f1c40f')
        ax1.axhspan(0.85, 0.90, alpha=0.1, color='#e67e22')
        ax1.axhspan(0, 0.85, alpha=0.1, color='#e74c3c')

        # 动态Y轴范围（放大差异）
        y_range = means.max() - means.min()
        y_padding = max(y_range * 0.3, 0.02)
        y_min = max(min(means.min() - y_padding, means.min() - stds.max()), 0.70)
        y_max = min(max(means.max() + y_padding, 0.98), 1.0)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlim(-0.5, n_files - 0.5)

        ax1.set_xlabel('文件序号（模型轮数递增 →）', fontsize=11)
        ax1.set_ylabel('连续性值', fontsize=11)
        ax1.set_title('📈 平均帧间连续性趋势', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax1.grid(True, alpha=0.4, linestyle='-', zorder=1)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(20, n_files)))

        # ==================== 图2: 最差帧检测 ====================
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(x, mins, 'r-s', markersize=marker_size,
                 linewidth=1.2, label='最小值（最差帧）', alpha=0.9, zorder=3)
        ax2.plot(x, percentile_5, color='#e67e22', marker='^',
                 markersize=marker_size, linewidth=1.2,
                 label='第5百分位数', alpha=0.9, zorder=3)

        ax2.axhline(y=0.85, color='#e74c3c', linestyle='--', linewidth=2)
        ax2.axhline(y=0.90, color='#f39c12', linestyle='--', linewidth=2)

        # 动态Y轴
        y_min2 = max(min(mins.min() - 0.08, percentile_5.min() - 0.05), 0.3)
        ax2.set_ylim(y_min2, 1.0)
        ax2.set_xlim(-0.5, n_files - 0.5)

        ax2.set_xlabel('文件序号（模型轮数递增 →）', fontsize=11)
        ax2.set_ylabel('连续性值', fontsize=11)
        ax2.set_title('🔍 最差帧连续性（断裂检测）', fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax2.grid(True, alpha=0.4)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(20, n_files)))

        # ==================== 图3: 问题帧比例堆叠图 ====================
        ax3 = fig.add_subplot(gs[1, 0])

        bar_width = 0.85
        ax3.bar(x, below_085, bar_width, label='< 0.85（连贯性差）',
                color='#e74c3c', alpha=0.9)
        ax3.bar(x, below_090 - below_085, bar_width, bottom=below_085,
                label='0.85-0.90（轻微断裂）', color='#f39c12', alpha=0.9)
        ax3.bar(x, below_095 - below_090, bar_width, bottom=below_090,
                label='0.90-0.95（正常偏低）', color='#f1c40f', alpha=0.9)

        ax3.set_xlabel('文件序号（模型轮数递增 →）', fontsize=11)
        ax3.set_ylabel('帧比例 (%)', fontsize=11)
        ax3.set_title('📊 低于阈值的帧比例分布', fontsize=13, fontweight='bold', pad=10)
        ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
        ax3.grid(True, alpha=0.4, axis='y')
        ax3.set_xlim(-0.5, n_files - 0.5)
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(20, n_files)))

        # ==================== 图4: 综合评分趋势 ====================
        ax4 = fig.add_subplot(gs[1, 1])

        bar_colors = [get_quality_color(m) for m in means]
        bars = ax4.bar(x, scores, color=bar_colors, alpha=0.85, width=bar_width)

        # 添加趋势线
        z = np.polyfit(x, scores, 1)
        p = np.poly1d(z)
        ax4.plot(x, p(x), 'b--', linewidth=2, alpha=0.7, label=f'趋势线 (斜率: {z[0]:.2f})')

        ax4.axhline(y=80, color='#27ae60', linestyle='--', linewidth=1.5, label='优秀 (≥80)')
        ax4.axhline(y=60, color='#f1c40f', linestyle='--', linewidth=1.5, label='良好 (≥60)')
        ax4.axhline(y=40, color='#e67e22', linestyle='--', linewidth=1.5, label='一般 (≥40)')

        ax4.set_ylim(0, 105)
        ax4.set_xlim(-0.5, n_files - 0.5)
        ax4.set_xlabel('文件序号（模型轮数递增 →）', fontsize=11)
        ax4.set_ylabel('综合评分 (0-100)', fontsize=11)
        ax4.set_title('🏆 连续性综合评分', fontsize=13, fontweight='bold', pad=10)
        ax4.legend(loc='lower right', fontsize=9, framealpha=0.95)
        ax4.grid(True, alpha=0.4, axis='y')
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(20, n_files)))

        # ==================== 图5: 文件详情水平条形图 ====================
        ax5 = fig.add_subplot(gs[2, :])

        # 按平均值降序排序
        sorted_indices = np.argsort(means)[::-1]
        sorted_names = [filenames[i] for i in sorted_indices]
        sorted_means = means[sorted_indices]
        sorted_scores = scores[sorted_indices]
        sorted_indices_original = [valid_results[i]['index'] for i in sorted_indices]

        # 颜色编码
        bar_colors = [get_quality_color(m) for m in sorted_means]

        y_pos = np.arange(len(sorted_names))
        bars = ax5.barh(y_pos, sorted_means, color=bar_colors, height=0.78, alpha=0.9)

        # 在条形末端显示数值和序号
        for i, (bar, score, orig_idx) in enumerate(zip(bars, sorted_scores, sorted_indices_original)):
            width = bar.get_width()
            ax5.text(width + 0.003, bar.get_y() + bar.get_height() / 2,
                     f'{width:.4f} [评分:{score:.0f}] #{orig_idx + 1}',
                     ha='left', va='center', fontsize=max(6, 9 - n_files // 20))

        # Y轴标签（文件名）
        ax5.set_yticks(y_pos)
        fontsize_y = max(5, min(9, 11 - n_files // 12))
        ax5.set_yticklabels(sorted_names, fontsize=fontsize_y)

        # 阈值线
        ax5.axvline(x=0.95, color='#27ae60', linestyle='--', linewidth=2, label='0.95')
        ax5.axvline(x=0.90, color='#f39c12', linestyle='--', linewidth=2, label='0.90')
        ax5.axvline(x=0.85, color='#e74c3c', linestyle='--', linewidth=2, label='0.85')

        # X轴范围（放大差异）
        x_range = sorted_means.max() - sorted_means.min()
        x_padding = max(x_range * 0.2, 0.01)
        x_min = max(sorted_means.min() - x_padding, 0.65)
        x_max = min(sorted_means.max() + x_padding + 0.06, 1.05)
        ax5.set_xlim(x_min, x_max)

        ax5.set_xlabel('平均连续性值', fontsize=11)
        ax5.set_title('📋 各文件平均连续性详情（按值降序 | #序号表示原始顺序）',
                      fontsize=13, fontweight='bold', pad=10)
        ax5.grid(True, alpha=0.4, axis='x')

        # 图例
        legend_elements = [
            Patch(facecolor='#27ae60', label='≥0.95 非常连贯'),
            Patch(facecolor='#2ecc71', label='0.90-0.95 正常'),
            Patch(facecolor='#f39c12', label='0.85-0.90 轻微断裂'),
            Patch(facecolor='#e74c3c', label='<0.85 连贯性差')
        ]
        ax5.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

        # ==================== 添加说明文本（背景透明）====================
        info_text = (
            "【帧间连续性指标 Frame-level Continuity】\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "计算公式: Cos(t,t+1) = [向量(t)·向量(t+1)] / [||t||×||t+1||]\n"
            "特征提取: MFCC (13维梅尔频率倒谱系数)\n"
            "检测目标: AI翻唱中的声音断裂、不连贯问题\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "阈值参考:\n"
            "  ● ≥0.95: 非常连贯（优秀）\n"
            "  ● 0.90-0.95: 正常（良好）\n"
            "  ● 0.85-0.90: 轻微断裂（需关注）\n"
            "  ● <0.85: 连贯性差（需优化模型）\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"本次分析: {n_files}个文件\n"
            f"均值范围: {means.min():.4f} ~ {means.max():.4f}\n"
            f"整体均值: {means.mean():.4f}"
        )

        # 背景透明的文本框
        fig.text(0.005, 0.995, info_text, fontsize=9,
                 verticalalignment='top', horizontalalignment='left',
                 transform=fig.transFigure,
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='none',  # 透明背景
                           edgecolor='#888888',
                           linewidth=1))

        # 总标题
        fig.suptitle('🎵 AI翻唱音频质量评估 - 帧间连续性分析',
                     fontsize=18, fontweight='bold', y=0.98)

        plt.show()

    # ==================== 主逻辑 ====================

    if not wav_paths:
        print("❌ 错误：文件路径列表为空")
        return None

    print(f"\n{'=' * 70}")
    print(f"🎵 帧间连续性分析 - 开始处理 {len(wav_paths)} 个WAV文件")
    print(f"{'=' * 70}\n")

    # 准备任务
    tasks = [(i, path) for i, path in enumerate(wav_paths)]

    # 并发处理
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=min(8, len(wav_paths))) as executor:
        futures = {executor.submit(compute_single_file, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            if result['success']:
                status = "✅"
                detail = f"均值={result['mean']:.4f}, 最小={result['min']:.4f}"
            else:
                status = "❌"
                detail = f"错误: {result.get('error', 'Unknown')}"

            # 截断过长的文件名
            name_display = result['filename'][:45]
            if len(result['filename']) > 45:
                name_display += "..."

            print(f"[{completed:3d}/{len(wav_paths)}] {status} {name_display:<50} {detail}")

    # 按原始顺序排序
    results.sort(key=lambda x: x['index'])

    # 过滤结果
    valid_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]

    print(f"\n{'=' * 70}")
    print(f"📊 分析完成！成功: {len(valid_results)}, 失败: {len(failed_results)}")
    print(f"{'=' * 70}")

    if failed_results:
        print("\n⚠️  失败的文件:")
        for r in failed_results:
            print(f"   • {r['filename']}: {r.get('error', 'Unknown error')}")

    if not valid_results:
        print("\n❌ 没有可用的分析结果！")
        return None

    # 统计摘要
    means = np.array([r['mean'] for r in valid_results])
    mins = np.array([r['min'] for r in valid_results])

    best_idx = np.argmax(means)
    worst_idx = np.argmin(means)

    print(f"\n{'─' * 50}")
    print(f"📈 统计摘要")
    print(f"{'─' * 50}")
    print(f"  • 平均连续性范围: {means.min():.4f} ~ {means.max():.4f}")
    print(f"  • 整体平均值: {means.mean():.4f} ± {means.std():.4f}")
    print(f"  • 最差帧范围: {mins.min():.4f} ~ {mins.max():.4f}")
    print(f"{'─' * 50}")
    print(f"  🏆 最佳: {valid_results[best_idx]['filename']}")
    print(f"          均值={means[best_idx]:.4f}, 最小={mins[best_idx]:.4f}")
    print(f"  ⚠️  最差: {valid_results[worst_idx]['filename']}")
    print(f"          均值={means[worst_idx]:.4f}, 最小={mins[worst_idx]:.4f}")
    print(f"{'─' * 50}\n")

    # 质量分布统计
    excellent = np.sum(means >= 0.95)
    good = np.sum((means >= 0.90) & (means < 0.95))
    fair = np.sum((means >= 0.85) & (means < 0.90))
    poor = np.sum(means < 0.85)

    print(f"📊 质量分布:")
    print(f"  • 非常连贯 (≥0.95): {excellent} 个 ({excellent / len(means) * 100:.1f}%)")
    print(f"  • 正常 (0.90-0.95): {good} 个 ({good / len(means) * 100:.1f}%)")
    print(f"  • 轻微断裂 (0.85-0.90): {fair} 个 ({fair / len(means) * 100:.1f}%)")
    print(f"  • 连贯性差 (<0.85): {poor} 个 ({poor / len(means) * 100:.1f}%)")

    # 创建可视化
    print(f"\n🎨 正在生成可视化图表...")
    create_visualization(valid_results)

    return valid_results


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import glob

    # 示例用法（请根据实际情况修改路径）

    # 方式1: 直接传入文件列表
    # wav_files = [
    #     r"D:\models\epoch_100.wav",
    #     r"D:\models\epoch_200.wav",
    #     r"D:\models\epoch_300.wav",
    # ]

    # 方式2: 使用glob批量获取并排序
    # wav_files = sorted(glob.glob(r"D:\ai_covers\*.wav"))

    # 方式3: 按特定规则排序
    # wav_files = sorted(glob.glob(r"D:\models\*.wav"),
    #                    key=lambda x: int(Path(x).stem.split('_')[-1]))

    # 调用分析函数
    # results = analyze_frame_continuity(wav_files)

    print("\n" + "=" * 60)
    print("帧间连续性分析工具")
    print("=" * 60)
    print("\n使用方法:")
    print("  from continuity_analyzer import analyze_frame_continuity")
    print("  results = analyze_frame_continuity(wav_file_list)")
    print("\n参数说明:")
    print("  wav_file_list: WAV文件路径的字符串列表（已按模型轮数排序）")