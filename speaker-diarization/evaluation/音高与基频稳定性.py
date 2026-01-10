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
11. 文字描述不能遮住图表本身,将文字描述的背景颜色设置为透明，图表使用常规字体而不是等宽字体
12. 尽量将代码都收敛到函数内部，方便调用，按文件进行并发处理，提升处理速度
13. 最后提供一个完整可用的python函数，以及其需要安装的依赖

+ 音高与基频稳定性（Pitch / F0）
    + 含义：F0曲线是否平滑连续，无频繁断裂/跳变；使用PYIN或CREPE算法提取，评估模型/声码器稳定性（如锯齿状或死点表示AI
      artifact）。
    + 平滑、少自然抖动：正常歌声
    + 锯齿/断层明显：AI抖动或不稳
    + 在长音处应有自然颤音（Vibrato），转音处平滑过渡。
"""

# pip install numpy librosa matplotlib scipy


"""
AI翻唱音频 F0（音高/基频）稳定性分析工具

依赖安装:
pip install numpy librosa matplotlib scipy

使用方法:
from f0_analyzer import analyze_f0_stability
analyze_f0_stability(["path1.wav", "path2.wav", ...])
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import warnings

warnings.filterwarnings('ignore')


def _extract_f0_features(wav_path: str, sr: int = 22050) -> dict:
    """
    提取单个文件的F0特征（内部函数）

    Args:
        wav_path: wav文件路径
        sr: 采样率

    Returns:
        包含F0特征的字典
    """
    import librosa

    try:
        # 加载音频
        y, sr = librosa.load(wav_path, sr=sr)

        # 使用PYIN算法提取F0（比YIN更鲁棒）
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),  # 约65Hz
            fmax=librosa.note_to_hz('C7'),  # 约2093Hz
            sr=sr,
            frame_length=2048,
            hop_length=512
        )

        # 获取有效的F0值（非NaN）
        valid_mask = ~np.isnan(f0)
        valid_f0 = f0[valid_mask]

        if len(valid_f0) < 20:
            return {
                'path': wav_path,
                'name': Path(wav_path).stem,
                'error': '有效F0帧数不足'
            }

        # ==================== 计算各项指标 ====================

        # 1. F0变化平滑度（一阶差分的标准差，单位：Hz）
        #    越小表示F0曲线越平滑
        f0_diff = np.diff(valid_f0)
        smoothness = np.std(f0_diff)

        # 2. 跳变率（F0变化超过阈值的帧占比）
        #    使用相对阈值：变化超过当前F0的5%视为跳变
        relative_diff = np.abs(f0_diff) / valid_f0[:-1]
        jump_threshold = 0.05  # 5%
        jump_count = np.sum(relative_diff > jump_threshold)
        jump_rate = jump_count / len(f0_diff) * 100  # 百分比

        # 3. Jitter（抖动）- 相邻帧F0变化的平均值
        #    适度的抖动表示自然的人声颤音
        jitter = np.mean(np.abs(f0_diff))

        # 4. Jitter百分比（相对抖动）
        jitter_percent = np.mean(np.abs(f0_diff) / valid_f0[:-1]) * 100

        # 5. 有效F0占比（检测到清晰基频的帧比例）
        valid_ratio = np.sum(valid_mask) / len(f0) * 100  # 百分比

        # 6. F0范围（最高与最低的差值，单位：半音）
        f0_range_semitones = 12 * np.log2(np.max(valid_f0) / np.min(valid_f0))

        # 7. 断裂次数（连续NaN区域的数量，表示AI无法生成的部分）
        nan_mask = np.isnan(f0)
        nan_diff = np.diff(nan_mask.astype(int))
        break_count = np.sum(nan_diff == 1)  # 从有效变为无效的次数

        # 8. 综合稳定性得分（0-100，越高越稳定）
        #    基于多个指标的加权计算
        stability_score = 100 - (
                min(smoothness / 10, 30) +  # 平滑度惩罚
                min(jump_rate * 2, 30) +  # 跳变率惩罚
                min(jitter_percent * 5, 20) +  # 抖动惩罚
                min((100 - valid_ratio) * 0.5, 20)  # 有效率惩罚
        )
        stability_score = max(0, min(100, stability_score))

        return {
            'path': wav_path,
            'name': Path(wav_path).stem,
            'smoothness': smoothness,  # Hz，越小越好
            'jump_rate': jump_rate,  # %，越小越好
            'jitter': jitter,  # Hz
            'jitter_percent': jitter_percent,  # %
            'valid_ratio': valid_ratio,  # %，越高越好
            'f0_range': f0_range_semitones,  # 半音
            'break_count': break_count,  # 次数，越少越好
            'stability_score': stability_score,  # 综合得分
            'f0_curve': f0,  # 原始F0曲线
            'mean_f0': np.mean(valid_f0),  # 平均F0
            'error': None
        }

    except Exception as e:
        return {
            'path': wav_path,
            'name': Path(wav_path).stem,
            'error': str(e)
        }


def _setup_chinese_font():
    """设置中文字体"""
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB',
        'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Sans CN',
        'Arial Unicode MS', 'STHeiti'
    ]

    available_fonts = [f.name for f in font_manager.fontManager.ttflist]

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = [font, 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return font

    # 如果没有中文字体，使用默认字体
    plt.rcParams['font.family'] = 'sans-serif'
    return None


def analyze_f0_stability(wav_paths: list, max_workers: int = None):
    """
    分析多个wav文件的音高与基频稳定性（F0）

    Args:
        wav_paths: wav文件路径的字符串数组（已按模型轮数排序）
        max_workers: 并发处理的最大进程数，默认为CPU核心数

    Returns:
        包含所有分析结果的列表
    """

    if not wav_paths:
        print("❌ 错误：文件路径列表为空")
        return None

    print(f"📊 开始分析 {len(wav_paths)} 个音频文件的F0稳定性...")
    print("=" * 60)

    # ==================== 并发处理 ====================
    results = []
    failed_files = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_extract_f0_features, path): path
            for path in wav_paths
        }

        for i, future in enumerate(as_completed(future_to_path)):
            path = future_to_path[future]
            try:
                result = future.result()
                if result['error']:
                    failed_files.append((path, result['error']))
                    print(f"  ⚠️  [{i + 1}/{len(wav_paths)}] {Path(path).name}: {result['error']}")
                else:
                    results.append(result)
                    print(f"  ✅ [{i + 1}/{len(wav_paths)}] {Path(path).name}")
            except Exception as e:
                failed_files.append((path, str(e)))
                print(f"  ❌ [{i + 1}/{len(wav_paths)}] {Path(path).name}: {e}")

    if not results:
        print("\n❌ 没有成功处理的文件")
        return None

    # 按原始顺序排序结果
    path_order = {path: i for i, path in enumerate(wav_paths)}
    results.sort(key=lambda x: path_order.get(x['path'], float('inf')))

    print(f"\n✅ 成功处理 {len(results)} 个文件，失败 {len(failed_files)} 个")
    print("=" * 60)

    # ==================== 可视化 ====================
    _visualize_f0_results(results)

    return results


def _visualize_f0_results(results: list):
    """
    可视化F0分析结果
    """
    # 设置中文字体
    font_name = _setup_chinese_font()

    n_files = len(results)

    # 提取数据
    names = [r['name'] for r in results]
    # 简化文件名显示（如果太长）
    short_names = []
    for i, name in enumerate(names):
        if len(name) > 15:
            short_name = name[:7] + "..." + name[-5:]
        else:
            short_name = name
        short_names.append(f"{i + 1}.{short_name}")

    smoothness = [r['smoothness'] for r in results]
    jump_rate = [r['jump_rate'] for r in results]
    jitter_percent = [r['jitter_percent'] for r in results]
    valid_ratio = [r['valid_ratio'] for r in results]
    stability_score = [r['stability_score'] for r in results]
    break_count = [r['break_count'] for r in results]

    # ==================== 创建图表 ====================

    # 根据文件数量调整图表大小
    fig_width = max(14, min(24, n_files * 0.3))
    fig_height = 16

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle('AI翻唱音频 F0（音高/基频）稳定性分析报告',
                 fontsize=16, fontweight='bold', y=0.98)

    # 创建网格布局
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)

    x = np.arange(n_files)

    # 颜色映射（根据稳定性得分）
    colors = plt.cm.RdYlGn(np.array(stability_score) / 100)

    # ---------- 图1: 综合稳定性得分 ----------
    ax1 = fig.add_subplot(gs[0, :])
    bars1 = ax1.bar(x, stability_score, color=colors, edgecolor='gray', linewidth=0.5)
    ax1.axhline(y=70, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='良好阈值 (70)')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='警告阈值 (50)')
    ax1.set_ylabel('得分', fontsize=11)
    ax1.set_title('📈 综合稳定性得分 (0-100，越高越好)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax1.grid(axis='y', alpha=0.3)

    # 添加趋势线
    z = np.polyfit(x, stability_score, 1)
    p = np.poly1d(z)
    ax1.plot(x, p(x), "b--", alpha=0.5, linewidth=2, label='趋势线')

    # 在柱状图上显示数值
    for i, (bar, score) in enumerate(zip(bars1, stability_score)):
        height = bar.get_height()
        ax1.annotate(f'{score:.0f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=6,
                     color='black' if score > 30 else 'white')

    # 添加说明文字（透明背景）
    desc_text = ('指标说明：综合考虑F0平滑度、跳变率、抖动和有效率\n'
                 '• ≥70: 优秀 (绿色)  • 50-70: 一般 (黄色)  • <50: 较差 (红色)')
    ax1.text(0.02, 0.95, desc_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white',
                                                alpha=0.7, edgecolor='gray'))

    # ---------- 图2: F0平滑度 ----------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, smoothness, 'o-', color='steelblue', linewidth=1.5, markersize=4)
    ax2.fill_between(x, smoothness, alpha=0.3, color='steelblue')
    ax2.set_ylabel('标准差 (Hz)', fontsize=10)
    ax2.set_title('🎵 F0变化平滑度（一阶差分标准差）', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)
    ax2.grid(alpha=0.3)

    # 动态调整Y轴范围
    y_min, y_max = min(smoothness), max(smoothness)
    y_padding = (y_max - y_min) * 0.15
    ax2.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

    # 标记最佳和最差
    best_idx = np.argmin(smoothness)
    worst_idx = np.argmax(smoothness)
    ax2.scatter([best_idx], [smoothness[best_idx]], color='green', s=100, zorder=5, marker='*')
    ax2.scatter([worst_idx], [smoothness[worst_idx]], color='red', s=100, zorder=5, marker='*')

    desc_text2 = '含义：F0曲线变化的剧烈程度\n• 越小越平滑自然\n• 过大表示AI抖动/锯齿'
    ax2.text(0.98, 0.95, desc_text2, transform=ax2.transAxes, fontsize=7,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    # ---------- 图3: 跳变率 ----------
    ax3 = fig.add_subplot(gs[1, 1])
    bars3 = ax3.bar(x, jump_rate, color='coral', edgecolor='gray', linewidth=0.5, alpha=0.8)
    ax3.axhline(y=5, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='警告阈值 (5%)')
    ax3.set_ylabel('百分比 (%)', fontsize=10)
    ax3.set_title('⚡ F0跳变率（突变帧占比）', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)

    desc_text3 = '含义：F0变化超过5%的帧占比\n• 越低越稳定\n• 高跳变率=音高不稳定'
    ax3.text(0.02, 0.95, desc_text3, transform=ax3.transAxes, fontsize=7,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    # ---------- 图4: 抖动百分比 ----------
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(x, jitter_percent, 's-', color='purple', linewidth=1.5, markersize=4)
    ax4.fill_between(x, jitter_percent, alpha=0.2, color='purple')

    # 标记理想范围
    ax4.axhspan(0.5, 2.0, alpha=0.15, color='green', label='理想范围 (0.5-2%)')
    ax4.set_ylabel('百分比 (%)', fontsize=10)
    ax4.set_title('🎤 Jitter抖动率（相对频率波动）', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)
    ax4.grid(alpha=0.3)
    ax4.legend(loc='upper right', fontsize=8)

    desc_text4 = '含义：相邻帧F0变化的平均比例\n• 0.5-2%: 自然颤音\n• 过低: 机械感  过高: 不稳定'
    ax4.text(0.98, 0.95, desc_text4, transform=ax4.transAxes, fontsize=7,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    # ---------- 图5: 有效F0占比 ----------
    ax5 = fig.add_subplot(gs[2, 1])
    bars5 = ax5.bar(x, valid_ratio, color='seagreen', edgecolor='gray', linewidth=0.5, alpha=0.8)
    ax5.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='良好阈值 (80%)')
    ax5.set_ylabel('百分比 (%)', fontsize=10)
    ax5.set_title('✅ 有效F0检测率', fontsize=11, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)
    ax5.set_ylim(0, 105)
    ax5.grid(axis='y', alpha=0.3)
    ax5.legend(loc='lower right', fontsize=8)

    desc_text5 = '含义：成功检测到清晰基频的帧比例\n• 越高越好\n• 低比例=声音模糊/断裂'
    ax5.text(0.02, 0.25, desc_text5, transform=ax5.transAxes, fontsize=7,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))

    # ==================== 添加统计摘要 ====================

    # 在图表底部添加汇总信息
    summary_text = (
        f"📊 统计摘要  |  "
        f"文件总数: {n_files}  |  "
        f"平均稳定性得分: {np.mean(stability_score):.1f}  |  "
        f"最佳: {names[np.argmax(stability_score)]} ({max(stability_score):.0f}分)  |  "
        f"最差: {names[np.argmin(stability_score)]} ({min(stability_score):.0f}分)"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange'))

    # ==================== 显示图表 ====================
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # 使用TkAgg后端确保弹出窗口
    manager = plt.get_current_fig_manager()
    try:
        manager.window.state('zoomed')  # Windows最大化
    except:
        try:
            manager.resize(*manager.window.maxsize())  # Linux
        except:
            pass

    plt.show()

    # ==================== 打印详细报告 ====================
    print("\n" + "=" * 80)
    print("📋 详细分析报告")
    print("=" * 80)
    print(f"{'序号':<4} {'文件名':<25} {'稳定性':<8} {'平滑度':<10} {'跳变率':<8} {'抖动%':<8} {'有效率':<8}")
    print("-" * 80)

    for i, r in enumerate(results):
        name = r['name'][:22] + "..." if len(r['name']) > 25 else r['name']
        print(f"{i + 1:<4} {name:<25} {r['stability_score']:<8.1f} "
              f"{r['smoothness']:<10.2f} {r['jump_rate']:<8.2f} "
              f"{r['jitter_percent']:<8.2f} {r['valid_ratio']:<8.1f}")

    print("=" * 80)
    print("\n🏆 排名 (按稳定性得分):")
    sorted_results = sorted(results, key=lambda x: x['stability_score'], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i + 1}. {r['name']} - {r['stability_score']:.1f}分")

    print("\n⚠️  需关注 (得分最低的5个):")
    for i, r in enumerate(sorted_results[-5:]):
        print(f"  {i + 1}. {r['name']} - {r['stability_score']:.1f}分")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    import sys

    # 示例用法
    example_paths = [
        "model_epoch_100.wav",
        "model_epoch_200.wav",
        "model_epoch_300.wav",
        # ... 更多文件
    ]

    if len(sys.argv) > 1:
        # 从命令行参数获取路径
        analyze_f0_stability(sys.argv[1:])
    else:
        print("使用方法:")
        print("  python f0_analyzer.py file1.wav file2.wav ...")
        print("\n或在Python中调用:")
        print("  from f0_analyzer import analyze_f0_stability")
        print("  analyze_f0_stability(['file1.wav', 'file2.wav', ...])")