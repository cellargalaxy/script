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

+ 零交叉率（Zero-Crossing Rate, ZCR）
    + 含义：每秒信号过零次数，公式：ZCR = (1/(2*(N-1))) × Σ |sgn(x_{i+1}) - sgn(x_i)|；反映噪声和高频振荡，异常高表示数字感。
    + 小于0.1：低噪声
    + 0.1–0.5：正常波动
    + 大于0.5：高噪声/AI artifact
    + 副歌段方差过大表示气息不稳。
"""

# pip install numpy librosa matplotlib scipy

"""
零交叉率(ZCR)质量分析工具

依赖安装：
pip install numpy librosa matplotlib scipy

使用方法：
analyze_zcr_quality(["path/to/file1.wav", "path/to/file2.wav", ...])
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import warnings


def analyze_zcr_quality(wav_paths: List[str], max_workers: int = 8) -> Optional[Dict]:
    """
    分析多个WAV文件的零交叉率(ZCR)，并可视化对比

    参数:
        wav_paths: WAV文件路径的字符串数组（按模型轮数递增排序）
        max_workers: 并发处理的最大线程数

    返回:
        包含分析结果的字典，如果失败返回None
    """

    # ==================== 内部函数定义 ====================

    def _setup_matplotlib():
        """配置matplotlib使用常规中文字体"""
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'PingFang SC',
            'Hiragino Sans GB', 'Arial Unicode MS', 'sans-serif'
        ]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        warnings.filterwarnings('ignore')

    def _calculate_zcr_single(wav_path: str) -> Dict:
        """计算单个文件的ZCR指标（内部函数）"""
        import librosa

        try:
            # 加载音频
            y, sr = librosa.load(wav_path, sr=None)

            # 计算帧级ZCR
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]

            # 计算每秒的ZCR（用于更直观的阈值判断）
            # ZCR返回的是每帧中过零的比例，需要转换为每秒次数
            frames_per_second = sr / 512
            zcr_per_second = zcr * 2048 * frames_per_second / sr

            return {
                'path': wav_path,
                'filename': Path(wav_path).stem,
                'zcr_mean': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr)),
                'zcr_max': float(np.max(zcr)),
                'zcr_min': float(np.min(zcr)),
                'zcr_variance': float(np.var(zcr)),
                'zcr_median': float(np.median(zcr)),
                'zcr_q25': float(np.percentile(zcr, 25)),
                'zcr_q75': float(np.percentile(zcr, 75)),
                'duration': len(y) / sr,
                'sample_rate': sr,
                'success': True
            }
        except Exception as e:
            return {
                'path': wav_path,
                'filename': Path(wav_path).stem,
                'success': False,
                'error': str(e)
            }

    def _get_quality_color(zcr_value: float) -> str:
        """根据ZCR值返回对应的颜色"""
        if zcr_value < 0.1:
            return '#2ecc71'  # 绿色 - 优秀
        elif zcr_value <= 0.5:
            return '#f39c12'  # 橙色 - 正常
        else:
            return '#e74c3c'  # 红色 - 高噪声

    def _get_quality_label(zcr_value: float) -> str:
        """根据ZCR值返回质量标签"""
        if zcr_value < 0.1:
            return '低噪声(优秀)'
        elif zcr_value <= 0.5:
            return '正常波动'
        else:
            return '高噪声/AI artifact'

    def _create_visualization(valid_results: List[Dict], n_files: int):
        """创建可视化图表"""

        # 提取数据
        filenames = [r['filename'] for r in valid_results]
        zcr_means = np.array([r['zcr_mean'] for r in valid_results])
        zcr_stds = np.array([r['zcr_std'] for r in valid_results])
        zcr_variances = np.array([r['zcr_variance'] for r in valid_results])
        zcr_q25 = np.array([r['zcr_q25'] for r in valid_results])
        zcr_q75 = np.array([r['zcr_q75'] for r in valid_results])

        # 动态计算图表尺寸
        fig_width = min(24, max(16, n_files * 0.15))
        fig_height = 14

        fig = plt.figure(figsize=(fig_width, fig_height))

        # ============ 主标题和说明 ============
        title_text = '零交叉率(ZCR)质量分析报告'
        subtitle_text = ('【定义】ZCR = (1/(2×(N-1))) × Σ|sgn(x_{i+1}) - sgn(x_i)|，每秒信号过零次数\n'
                         '【含义】反映噪声和高频振荡，异常高表示数字感/AI痕迹')

        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
        fig.text(0.5, 0.94, subtitle_text, ha='center', fontsize=11,
                 style='italic', color='#555555',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none'))

        # ============ 图1: 趋势线图（核心图表） ============
        ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=1)

        x = np.arange(n_files)
        colors = [_get_quality_color(v) for v in zcr_means]

        # 绘制趋势线和误差范围
        ax1.fill_between(x, zcr_q25, zcr_q75, alpha=0.3, color='steelblue', label='四分位范围(Q25-Q75)')
        ax1.plot(x, zcr_means, 'b-', linewidth=2, label='平均ZCR', zorder=5)
        ax1.scatter(x, zcr_means, c=colors, s=50, edgecolors='white', linewidth=1, zorder=6)

        # 阈值参考线
        ax1.axhline(y=0.1, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.8, label='低噪声阈值 (0.1)')
        ax1.axhline(y=0.5, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8, label='高噪声阈值 (0.5)')

        # 质量区域填充
        y_min_plot = max(0, zcr_means.min() * 0.8 - 0.02)
        y_max_plot = min(1.0, max(zcr_means.max() * 1.2, 0.55))

        ax1.axhspan(y_min_plot, 0.1, alpha=0.08, color='green')
        ax1.axhspan(0.1, 0.5, alpha=0.08, color='yellow')
        ax1.axhspan(0.5, y_max_plot, alpha=0.08, color='red')

        ax1.set_xlim(-0.5, n_files - 0.5)
        ax1.set_ylim(y_min_plot, y_max_plot)

        # X轴标签处理（智能间隔）
        if n_files <= 25:
            ax1.set_xticks(x)
            ax1.set_xticklabels(filenames, rotation=45, ha='right', fontsize=7)
        else:
            step = max(1, n_files // 20)
            display_ticks = list(range(0, n_files, step))
            if (n_files - 1) not in display_ticks:
                display_ticks.append(n_files - 1)
            ax1.set_xticks(display_ticks)
            ax1.set_xticklabels([filenames[i] for i in display_ticks],
                                rotation=45, ha='right', fontsize=7)

        ax1.set_xlabel('文件 (按模型轮数递增 →)', fontsize=10)
        ax1.set_ylabel('ZCR值', fontsize=10)
        ax1.set_title('📈 ZCR随模型轮数变化趋势', fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # ============ 图2: 质量分布统计 ============
        ax2 = plt.subplot2grid((3, 4), (0, 3), rowspan=1)

        low_noise = int(np.sum(zcr_means < 0.1))
        normal = int(np.sum((zcr_means >= 0.1) & (zcr_means <= 0.5)))
        high_noise = int(np.sum(zcr_means > 0.5))

        categories = ['低噪声\n(<0.1)', '正常\n(0.1-0.5)', '高噪声\n(>0.5)']
        counts = [low_noise, normal, high_noise]
        bar_colors = ['#2ecc71', '#f39c12', '#e74c3c']

        bars = ax2.bar(categories, counts, color=bar_colors, edgecolor='white', linewidth=2)

        # 在柱子上显示数量和百分比
        for bar, count in zip(bars, counts):
            if count > 0:
                pct = count / n_files * 100
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f'{count}个\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

        ax2.set_ylabel('文件数量', fontsize=10)
        ax2.set_title('📊 质量分布统计', fontsize=12, fontweight='bold', pad=10)
        ax2.set_ylim(0, max(counts) * 1.3 if max(counts) > 0 else 1)

        # ============ 图3: 方差分析（气息稳定性） ============
        ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=3, rowspan=1)

        # 使用颜色编码质量
        bars = ax3.bar(x, zcr_variances, color=colors, alpha=0.75, edgecolor='white', linewidth=0.5)

        # 添加趋势线
        z = np.polyfit(x, zcr_variances, 3)
        p = np.poly1d(z)
        ax3.plot(x, p(x), 'b--', linewidth=2, alpha=0.7, label='趋势线')

        # 动态Y轴范围（放大差异）
        var_min, var_max = zcr_variances.min(), zcr_variances.max()
        var_range = var_max - var_min
        if var_range > 0:
            y_bottom = max(0, var_min - var_range * 0.15)
            y_top = var_max + var_range * 0.15
            ax3.set_ylim(y_bottom, y_top)

        ax3.set_xlim(-0.5, n_files - 0.5)

        # X轴标签
        if n_files <= 25:
            ax3.set_xticks(x)
            ax3.set_xticklabels(filenames, rotation=45, ha='right', fontsize=7)
        else:
            step = max(1, n_files // 20)
            display_ticks = list(range(0, n_files, step))
            if (n_files - 1) not in display_ticks:
                display_ticks.append(n_files - 1)
            ax3.set_xticks(display_ticks)
            ax3.set_xticklabels([filenames[i] for i in display_ticks],
                                rotation=45, ha='right', fontsize=7)

        ax3.set_xlabel('文件 (按模型轮数递增 →)', fontsize=10)
        ax3.set_ylabel('ZCR方差', fontsize=10)
        ax3.set_title('📉 ZCR方差分析（方差过大 = 气息不稳/副歌段异常）', fontsize=12, fontweight='bold', pad=10)
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # ============ 图4: 统计信息面板 ============
        ax4 = plt.subplot2grid((3, 4), (1, 3), rowspan=1)
        ax4.axis('off')

        # 找出关键文件
        best_idx = int(np.argmin(zcr_means))
        worst_idx = int(np.argmax(zcr_means))
        most_stable_idx = int(np.argmin(zcr_variances))
        least_stable_idx = int(np.argmax(zcr_variances))

        # 计算改进趋势
        if n_files >= 3:
            first_third = zcr_means[:n_files // 3].mean()
            last_third = zcr_means[-n_files // 3:].mean()
            improvement = ((first_third - last_third) / first_third * 100) if first_third != 0 else 0
            trend_text = f"{'↓ 改善' if improvement > 0 else '↑ 恶化'} {abs(improvement):.1f}%"
        else:
            trend_text = "样本不足"

        stats_text = f"""
┌─────────────────────────┐
│      📋 统计摘要        │
├─────────────────────────┤
│ 文件总数: {n_files:>14} │
│ 平均ZCR:  {zcr_means.mean():>14.4f} │
│ 最小ZCR:  {zcr_means.min():>14.4f} │
│ 最大ZCR:  {zcr_means.max():>14.4f} │
│ 训练趋势: {trend_text:>14} │
├─────────────────────────┤
│      🏆 最佳文件        │
│ {filenames[best_idx][:23]:^23} │
│ ZCR = {zcr_means[best_idx]:.4f} ({_get_quality_label(zcr_means[best_idx])})│
├─────────────────────────┤
│      ⚠️  最差文件        │
│ {filenames[worst_idx][:23]:^23} │
│ ZCR = {zcr_means[worst_idx]:.4f} ({_get_quality_label(zcr_means[worst_idx])})│
├─────────────────────────┤
│      🎯 最稳定          │
│ {filenames[most_stable_idx][:23]:^23} │
│ 方差 = {zcr_variances[most_stable_idx]:.6f}     │
└─────────────────────────┘
"""
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='sans-serif',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                           edgecolor='#dee2e6', alpha=0.95))

        # ============ 图5: 热力图概览 ============
        ax5 = plt.subplot2grid((3, 4), (2, 0), colspan=4, rowspan=1)

        # 创建热力图数据
        metrics_data = np.array([zcr_means, zcr_stds, zcr_variances])
        metrics_labels = ['平均ZCR', '标准差', '方差']

        # 归一化用于显示
        metrics_normalized = np.zeros_like(metrics_data)
        for i in range(3):
            min_val, max_val = metrics_data[i].min(), metrics_data[i].max()
            if max_val - min_val > 0:
                metrics_normalized[i] = (metrics_data[i] - min_val) / (max_val - min_val)
            else:
                metrics_normalized[i] = 0.5

        # 绘制热力图
        im = ax5.imshow(metrics_normalized, aspect='auto', cmap='RdYlGn_r',
                        interpolation='nearest', vmin=0, vmax=1)

        ax5.set_yticks(range(3))
        ax5.set_yticklabels(metrics_labels, fontsize=10)

        # X轴标签
        if n_files <= 30:
            ax5.set_xticks(x)
            ax5.set_xticklabels(filenames, rotation=90, ha='center', fontsize=6)
        else:
            step = max(1, n_files // 25)
            display_ticks = list(range(0, n_files, step))
            ax5.set_xticks(display_ticks)
            ax5.set_xticklabels([filenames[i] for i in display_ticks],
                                rotation=90, ha='center', fontsize=6)

        ax5.set_title('🗺️ 多指标热力图概览（绿色=优秀，红色=较差）', fontsize=12, fontweight='bold', pad=10)

        # 颜色条
        cbar = plt.colorbar(im, ax=ax5, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('相对值 (归一化)', fontsize=9)

        # ============ 底部说明文字 ============
        info_text = ('【阈值参考】 ZCR < 0.1: 低噪声(优秀)  |  0.1 ≤ ZCR ≤ 0.5: 正常波动  |  '
                     'ZCR > 0.5: 高噪声/AI artifact  |  方差过大: 气息不稳/副歌段异常')

        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                 style='italic', color='#666666',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='none', edgecolor='none'))

        plt.tight_layout(rect=[0, 0.04, 1, 0.92])
        plt.show()

    # ==================== 主处理逻辑 ====================

    _setup_matplotlib()

    if not wav_paths:
        print("❌ 错误: 文件路径列表为空")
        return None

    print(f"🔍 开始分析 {len(wav_paths)} 个WAV文件...")

    # 并发处理所有文件
    results = [None] * len(wav_paths)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(wav_paths))) as executor:
        future_to_idx = {
            executor.submit(_calculate_zcr_single, path): i
            for i, path in enumerate(wav_paths)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1

            # 进度显示
            if completed % 10 == 0 or completed == len(wav_paths):
                print(f"  ✓ 已完成: {completed}/{len(wav_paths)}")

    # 过滤成功的结果
    valid_results = [r for r in results if r and r.get('success', False)]
    failed_results = [r for r in results if r and not r.get('success', False)]

    if failed_results:
        print(f"\n⚠️ 警告: {len(failed_results)} 个文件处理失败:")
        for r in failed_results[:5]:  # 只显示前5个
            print(f"    - {r['filename']}: {r.get('error', 'Unknown error')}")
        if len(failed_results) > 5:
            print(f"    ... 还有 {len(failed_results) - 5} 个失败")

    if not valid_results:
        print("❌ 错误: 没有成功分析的文件")
        return None

    n_files = len(valid_results)
    print(f"\n✅ 成功分析 {n_files} 个文件，正在生成图表...")

    # 创建可视化
    _create_visualization(valid_results, n_files)

    # 返回分析结果
    return {
        'total_files': len(wav_paths),
        'successful': n_files,
        'failed': len(failed_results),
        'results': valid_results,
        'summary': {
            'mean_zcr': float(np.mean([r['zcr_mean'] for r in valid_results])),
            'best_file': valid_results[int(np.argmin([r['zcr_mean'] for r in valid_results]))]['filename'],
            'worst_file': valid_results[int(np.argmax([r['zcr_mean'] for r in valid_results]))]['filename'],
        }
    }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例用法
    import glob

    # 方式1: 手动指定文件列表
    # wav_files = [
    #     "path/to/model_epoch_100.wav",
    #     "path/to/model_epoch_200.wav",
    #     "path/to/model_epoch_300.wav",
    # ]

    # 方式2: 使用glob匹配
    # wav_files = sorted(glob.glob("path/to/outputs/*.wav"))

    # 调用分析函数
    # results = analyze_zcr_quality(wav_files)

    print("请提供WAV文件路径列表来运行分析")
    print("示例: results = analyze_zcr_quality(['file1.wav', 'file2.wav', ...])")