"""
1. 我打算使用以下指标对ai翻唱的wav文件进行质量评价。
2. 我只有使用ai翻唱出来的多个wav文件，我能提供这些文件的路径。
3. 判断以下指标，只有wav文件路径，这些文件之间是否能对比出优劣，如果对比不出优劣就不需要再继续了
4. 如果能对比出优劣，写一个python函数，入参是wav文件路径的字符串数组
5. 该python函数实现以下指标的计算，并且将计算结果画为图表进行可视化对比
6. 图表的类型，需要根据指标的特殊进行选择，目的是能更加直观的看出各个wav文件的优劣
7. 图表的数轴标度，为了避免不同文件之间的指标差异过小，在图中看不出区别，需要更加明显的处理
8. 文件大约有几十个，需要合理排版，以能清晰看出每个文件的数据走向与图标
9. 并且文件路径数组已经排好序，模型的轮数是递增的。
10. 在图表中增加该指标的文字描述，阈值的辅助信息，图表使用常规字体而不是等宽字体
11. 尽量将代码都收敛到函数内部，方便调用
12. 最后提供一个完整可用的python函数，以及其需要安装的依赖

+ 短时响度波动（Short-term Loudness Variance）
    + 含义：短时间窗（如3秒）内响度变化程度，反映情感表达的动态性；用于判断是否“全程一个音量”（情感死板）或压缩过度。
    + 适中（方差适度）：自然起伏，富有情感
    + 波动太小：情感死板
    + 波动太大：不稳定，破音风险
"""

# pip install numpy scipy matplotlib pyloudnorm


"""
短时响度波动分析工具 (Short-term Loudness Variance Analyzer)

依赖安装:
pip install numpy scipy matplotlib pyloudnorm

使用示例:
    wav_files = ["epoch_100.wav", "epoch_200.wav", "epoch_300.wav", ...]
    results = analyze_short_term_loudness_variance(wav_files)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

"""
短时响度波动分析工具 (Short-term Loudness Variance Analyzer)

依赖安装:
pip install numpy scipy matplotlib pyloudnorm

使用示例:
    wav_files = ["epoch_100.wav", "epoch_200.wav", "epoch_300.wav", ...]
    results = analyze_short_term_loudness_variance(wav_files)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

"""
短时响度波动分析工具 (Short-term Loudness Variance Analyzer)

依赖安装:
pip install numpy scipy matplotlib pyloudnorm

使用示例:
    wav_files = ["epoch_100.wav", "epoch_200.wav", "epoch_300.wav", ...]
    results = analyze_short_term_loudness_variance(wav_files)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def analyze_short_term_loudness_variance(
        wav_paths: list[str],
        window_sec: float = 3.0,
        hop_sec: float = 0.5,
        output_path: str = "loudness_variance_analysis.png"
) -> list[dict]:
    """
    分析多个WAV文件的短时响度波动，并生成可视化对比图表。

    参数:
        wav_paths: WAV文件路径的字符串数组（已按模型轮数排序）
        window_sec: 短时窗口长度（秒），默认3秒
        hop_sec: 窗口滑动步长（秒），默认0.5秒
        output_path: 图表保存路径

    返回:
        包含每个文件分析结果的字典列表
    """

    # ========================
    # 1. 导入并检查依赖
    # ========================
    try:
        import pyloudnorm as pyln
    except ImportError:
        raise ImportError(
            "缺少依赖库，请运行: pip install pyloudnorm"
        )

    # ========================
    # 2. 设置中文字体
    # ========================
    def setup_chinese_font():
        """设置支持中文的字体"""
        import matplotlib.font_manager as fm

        # 候选中文字体列表（按优先级排序）
        chinese_fonts = [
            'SimHei',  # Windows 黑体
            'Microsoft YaHei',  # Windows 微软雅黑
            'PingFang SC',  # macOS 苹方
            'Heiti SC',  # macOS 黑体
            'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
            'Noto Sans CJK SC',  # Google Noto 中文
            'Source Han Sans SC',  # 思源黑体
            'DejaVu Sans',  # 备用
        ]

        # 获取系统可用字体
        available_fonts = set(f.name for f in fm.fontManager.ttflist)

        # 查找第一个可用的中文字体
        selected_font = 'DejaVu Sans'  # 默认备选
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                break

        plt.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False

        return selected_font

    # ========================
    # 3. 内部辅助函数
    # ========================
    def load_audio(path: str) -> tuple[np.ndarray, int]:
        """加载WAV文件并归一化为 [-1, 1] 的单声道音频"""
        sr, audio = wavfile.read(path)

        # 转换为 float64 并归一化
        if audio.dtype == np.int16:
            audio = audio.astype(np.float64) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float64) / 2147483648.0
        elif audio.dtype == np.float32:
            audio = audio.astype(np.float64)
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float64) - 128) / 128.0

        # 立体声转单声道
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        return audio, sr

    def compute_short_term_loudness(
            audio: np.ndarray,
            sr: int,
            window_sec: float,
            hop_sec: float
    ) -> np.ndarray:
        """计算短时响度序列 (LUFS)"""
        meter = pyln.Meter(sr)
        window_samples = int(window_sec * sr)
        hop_samples = int(hop_sec * sr)

        loudness_values = []
        for start in range(0, len(audio) - window_samples + 1, hop_samples):
            segment = audio[start:start + window_samples]
            try:
                loudness = meter.integrated_loudness(segment)
                if np.isfinite(loudness):
                    loudness_values.append(loudness)
            except Exception:
                continue

        return np.array(loudness_values) if loudness_values else np.array([-70.0])

    def truncate_name(name: str, max_len: int = 20) -> str:
        """截断过长的文件名"""
        name = os.path.splitext(name)[0]  # 移除扩展名
        if len(name) > max_len:
            return name[:max_len - 2] + ".."
        return name

    # ========================
    # 4. 分析所有文件
    # ========================
    results = []
    print(f"正在分析 {len(wav_paths)} 个文件...")

    for i, path in enumerate(wav_paths):
        try:
            audio, sr = load_audio(path)
            loudness_seq = compute_short_term_loudness(audio, sr, window_sec, hop_sec)

            result = {
                'index': i + 1,
                'path': path,
                'name': os.path.basename(path),
                'variance': float(np.var(loudness_seq)),
                'std': float(np.std(loudness_seq)),
                'mean_loudness': float(np.mean(loudness_seq)),
                'min_loudness': float(np.min(loudness_seq)),
                'max_loudness': float(np.max(loudness_seq)),
                'loudness_seq': loudness_seq,
                'duration_sec': len(audio) / sr
            }
            results.append(result)
            print(f"  [{i + 1}/{len(wav_paths)}] {result['name']}: 标准差={result['std']:.2f} dB")

        except Exception as e:
            print(f"  [{i + 1}/{len(wav_paths)}] 错误处理 {path}: {e}")
            results.append({
                'index': i + 1,
                'path': path,
                'name': os.path.basename(path),
                'variance': np.nan,
                'std': np.nan,
                'mean_loudness': np.nan,
                'min_loudness': np.nan,
                'max_loudness': np.nan,
                'loudness_seq': np.array([]),
                'duration_sec': 0,
                'error': str(e)
            })

    # ========================
    # 5. 可视化
    # ========================

    # 设置中文字体
    used_font = setup_chinese_font()
    print(f"使用字体: {used_font}")

    n_files = len(results)
    valid_results = [r for r in results if np.isfinite(r['std'])]

    if not valid_results:
        print("没有有效的分析结果！")
        return results

    # 计算图表尺寸
    fig_width = max(14, n_files * 0.4)
    fig_height = 16

    fig = plt.figure(figsize=(fig_width, fig_height))

    # 定义阈值
    THRESHOLD_LOW = 2.0  # 低于此值：情感死板
    THRESHOLD_HIGH = 6.0  # 高于此值：不稳定

    # ---- 图1: 短时响度标准差趋势图 (主图) ----
    ax1 = fig.add_subplot(3, 1, 1)

    x = np.arange(n_files)
    stds = np.array([r['std'] for r in results])

    # 根据阈值着色
    colors = []
    for s in stds:
        if np.isnan(s):
            colors.append('gray')
        elif s < THRESHOLD_LOW:
            colors.append('#FF6B6B')  # 红色：太小
        elif s > THRESHOLD_HIGH:
            colors.append('#FFA500')  # 橙色：太大
        else:
            colors.append('#4ECDC4')  # 青色：适中

    # 柱状图
    bars = ax1.bar(x, stds, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    # 趋势线
    valid_mask = np.isfinite(stds)
    if np.sum(valid_mask) > 1:
        z = np.polyfit(x[valid_mask], stds[valid_mask], 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), '--', color='#2C3E50', linewidth=2, label='趋势线', alpha=0.7)

    # 理想范围阴影
    ax1.axhspan(THRESHOLD_LOW, THRESHOLD_HIGH, alpha=0.15, color='green',
                label=f'理想范围 ({THRESHOLD_LOW}-{THRESHOLD_HIGH} dB)')
    ax1.axhline(y=THRESHOLD_LOW, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=THRESHOLD_HIGH, color='green', linestyle='--', linewidth=1.5, alpha=0.8)

    # 动态调整Y轴范围以放大差异
    valid_stds = stds[valid_mask]
    if len(valid_stds) > 0:
        y_min = max(0, np.min(valid_stds) - 1.5)
        y_max = np.max(valid_stds) + 1.5
        # 确保理想范围可见
        y_min = min(y_min, THRESHOLD_LOW - 0.5)
        y_max = max(y_max, THRESHOLD_HIGH + 0.5)
        ax1.set_ylim(y_min, y_max)

    # X轴标签
    labels = [truncate_name(r['name'], 15) for r in results]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)

    # 在柱顶显示数值
    for i, (bar, std_val) in enumerate(zip(bars, stds)):
        if np.isfinite(std_val):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{std_val:.1f}', ha='center', va='bottom', fontsize=7, rotation=0)

    ax1.set_xlabel('文件（按训练轮数排序 →）', fontsize=11)
    ax1.set_ylabel('短时响度标准差 (dB)', fontsize=11)
    ax1.set_title('短时响度波动分析 (Short-term Loudness Variance)', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-0.5, n_files - 0.5)

    # 中文说明文字框 - 设置为透明背景
    description = (
        "【指标说明】\n"
        "短时响度波动：衡量短时间窗（3秒）内响度变化程度，反映情感表达的动态性 用于判断是否全程一个音量（情感死板）或压缩过度。\n"
        "【阈值判断】\n"
        f"  ✓ 适中 ({THRESHOLD_LOW}-{THRESHOLD_HIGH} dB)：自然起伏，富有情感   ✗ 过小 (< {THRESHOLD_LOW} dB)：情感死板，缺乏变化   ✗ 过大 (> {THRESHOLD_HIGH} dB)：不稳定，有破音风险\n"
        "【颜色含义】\n"
        "  ● 青色 = 理想范围   ● 红色 = 波动太小（情感死板）   ● 橙色 = 波动太大（不稳定）"
    )
    ax1.text(0.02, 0.97, description, transform=ax1.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='none',  # 透明背景
                       edgecolor='#DEE2E6', alpha=0.8),  # 保留边框但透明度更高
             linespacing=1.4)

    # ---- 图2: 响度动态范围对比 ----
    ax2 = fig.add_subplot(3, 1, 2)

    means = np.array([r['mean_loudness'] for r in results])
    mins = np.array([r['min_loudness'] for r in results])
    maxs = np.array([r['max_loudness'] for r in results])

    # 绘制范围（误差棒样式）
    for i, r in enumerate(results):
        if np.isfinite(r['mean_loudness']):
            ax2.plot([i, i], [r['min_loudness'], r['max_loudness']],
                     color='#3498DB', linewidth=2, alpha=0.6)
            ax2.scatter([i], [r['mean_loudness']], color='#E74C3C',
                        s=30, zorder=5, edgecolor='white', linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax2.set_xlabel('文件（按训练轮数排序 →）', fontsize=11)
    ax2.set_ylabel('响度 (LUFS)', fontsize=11)
    ax2.set_title('响度动态范围对比（最小值 - 平均值 - 最大值）', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xlim(-0.5, n_files - 0.5)

    # 图例
    ax2.plot([], [], color='#3498DB', linewidth=2, label='动态范围（最小-最大）')
    ax2.scatter([], [], color='#E74C3C', s=30, label='平均响度')
    ax2.legend(loc='upper right', fontsize=9)

    # 图2说明 - 设置为透明背景
    desc2 = (
        "【图表说明】\n"
        "蓝色线段表示响度的最小值到最大值范围 红点表示平均响度值 范围越大说明动态变化越丰富"
    )
    ax2.text(0.02, 0.97, desc2, transform=ax2.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='none',  # 透明背景
                       edgecolor='#F0E68C', alpha=0.7),  # 保留边框但透明度更高
             linespacing=1.3)

    # ---- 图3: 响度时序热力图 ----
    ax3 = fig.add_subplot(3, 1, 3)

    # 将所有响度序列对齐到相同长度
    max_len = max(len(r['loudness_seq']) for r in results if len(r['loudness_seq']) > 0)
    if max_len > 0:
        heatmap_data = np.full((n_files, max_len), np.nan)
        for i, r in enumerate(results):
            seq = r['loudness_seq']
            if len(seq) > 0:
                # 重采样到统一长度
                if len(seq) < max_len:
                    indices = np.linspace(0, len(seq) - 1, max_len).astype(int)
                    heatmap_data[i, :] = seq[indices]
                else:
                    heatmap_data[i, :] = seq[:max_len]

        # 绘制热力图
        im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r',
                        vmin=np.nanpercentile(heatmap_data, 5),
                        vmax=np.nanpercentile(heatmap_data, 95))

        # 设置标签
        ax3.set_yticks(np.arange(n_files))
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_xlabel('时间（归一化）', fontsize=11)
        ax3.set_ylabel('文件', fontsize=11)
        ax3.set_title('响度时序热力图（红色=响亮，蓝色=安静）',
                      fontsize=12, fontweight='bold')

        # 颜色条
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8, pad=0.02)
        cbar.set_label('响度 (LUFS)', fontsize=10)

    # 图3说明 - 设置为透明背景
    desc3 = (
        "【图表说明】\n"
        "热力图展示每个文件在整个时间轴上的响度变化\n"
        "颜色越红表示越响亮，越蓝表示越安静\n"
        "颜色变化丰富说明情感表达更有层次"
    )
    ax3.text(0.02, 0.95, desc3, transform=ax3.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='none',  # 透明背景
                       edgecolor='#A3E4D7', alpha=0.7),  # 保留边框但透明度更高
             linespacing=1.3)

    # ========================
    # 6. 保存和显示
    # ========================
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n图表已保存至: {output_path}")
    plt.show()

    # ========================
    # 7. 打印统计摘要
    # ========================
    print("\n" + "=" * 60)
    print("分析摘要")
    print("=" * 60)

    valid_stds = [r['std'] for r in results if np.isfinite(r['std'])]
    if valid_stds:
        ideal_count = sum(1 for s in valid_stds if THRESHOLD_LOW <= s <= THRESHOLD_HIGH)
        low_count = sum(1 for s in valid_stds if s < THRESHOLD_LOW)
        high_count = sum(1 for s in valid_stds if s > THRESHOLD_HIGH)

        print(f"文件总数: {n_files}")
        print(f"有效文件: {len(valid_stds)}")
        print(f"标准差范围: {min(valid_stds):.2f} - {max(valid_stds):.2f} dB")
        print(f"\n【质量分布】")
        print(f"  ✓ 理想范围 ({THRESHOLD_LOW}-{THRESHOLD_HIGH} dB): {ideal_count} 个文件")
        print(f"  ✗ 情感死板 (< {THRESHOLD_LOW} dB): {low_count} 个文件")
        print(f"  ✗ 不稳定 (> {THRESHOLD_HIGH} dB): {high_count} 个文件")

        # 找出最接近理想值的文件
        ideal_center = (THRESHOLD_LOW + THRESHOLD_HIGH) / 2
        best_file = min(valid_results, key=lambda r: abs(r['std'] - ideal_center))
        print(f"\n【推荐】最佳文件: {best_file['name']}")
        print(f"         标准差: {best_file['std']:.2f} dB (最接近理想中值 {ideal_center} dB)")

    print("=" * 60)

    return results


# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    import glob

    # 示例：获取所有wav文件（按文件名排序）
    # wav_files = sorted(glob.glob("/path/to/your/wav/files/*.wav"))

    # 或者手动指定文件列表（按训练轮数排序）
    wav_files = [
        "epoch_100.wav",
        "epoch_200.wav",
        "epoch_300.wav",
        # ... 更多文件
    ]

    # 运行分析
    # results = analyze_short_term_loudness_variance(wav_files)