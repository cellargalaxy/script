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

# pip install librosa numpy matplotlib scipy


import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import warnings

# 忽略librosa在某些极短音频下的警告
warnings.filterwarnings('ignore')


def _analyze_single_file(args):
    """
    单个文件处理函数（用于多进程调用）。
    计算音高不稳定性指标。
    """
    file_path, index = args
    try:
        # 1. 加载音频 (sr=16000 提升速度，足够用于F0分析)
        y, sr = librosa.load(file_path, sr=16000)

        # 2. 提取基频 (F0) 使用 PYIN
        # frame_length=2048 保证低频分辨率
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=2048
        )

        if np.all(np.isnan(f0)):
            return index, os.path.basename(file_path), None

        # 3. 计算指标
        voiced_f0 = f0[~np.isnan(f0)]
        if len(voiced_f0) < 10:
            return index, os.path.basename(file_path), None

        # Hz -> Cents (音分) 转换，更符合人耳对音高的感知
        f0_cents = 1200 * np.log2(voiced_f0 / librosa.note_to_hz('C2'))

        # 计算一阶差分（相邻帧跳变）的绝对值
        abs_delta = np.abs(np.diff(f0_cents))

        # 计算平均值作为“不稳定性得分”
        instability_score = np.mean(abs_delta)

        return index, os.path.basename(file_path), instability_score

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return index, os.path.basename(file_path), None


def evaluate_pitch_stability_and_show(file_paths):
    """
    对wav文件进行音高稳定性评价，并弹出窗口显示图表。

    参数:
        file_paths (list): 按模型轮数排序好的wav文件路径字符串数组。
    """
    if not file_paths:
        print("文件列表为空。")
        return

    print(f"开始分析 {len(file_paths)} 个音频文件，正在并行计算 F0 指标...")

    # 准备多进程参数
    tasks = [(fp, i) for i, fp in enumerate(file_paths)]

    results = []
    # 并发处理
    with ProcessPoolExecutor() as executor:
        for result in executor.map(_analyze_single_file, tasks):
            results.append(result)
            if len(results) % 10 == 0:
                print(f"进度: {len(results)}/{len(file_paths)}...")

    # 排序与数据清洗
    results.sort(key=lambda x: x[0])
    valid_results = [r for r in results if r[2] is not None]
    filenames = [r[1] for r in valid_results]
    scores = [r[2] for r in valid_results]

    if not scores:
        print("未能提取到有效数据，无法绘图。")
        return

    print("分析完成，正在生成图表窗口...")

    # --- 绘图逻辑 ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # 字体设置
    font_options = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC', 'sans-serif']
    plt.rcParams['font.sans-serif'] = font_options
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布
    fig_width = max(12, len(filenames) * 0.2)
    fig, ax = plt.subplots(figsize=(fig_width, 8))  # 稍微调高一点高度

    x_indices = range(len(filenames))

    # 1. 绘制主数据线
    ax.plot(x_indices, scores, color='#2878B5', linewidth=2, marker='o', markersize=4,
            label='音高不稳定性 (Cents/Frame)')

    # 2. 绘制趋势线 (3次多项式拟合)
    if len(scores) > 5:
        z = np.polyfit(x_indices, scores, 3)
        p = np.poly1d(z)
        ax.plot(x_indices, p(x_indices), "r--", alpha=0.6, linewidth=1.5, label='训练总体趋势')

    # 3. 设置X轴
    ax.set_xlim(-0.5, len(filenames) - 0.5)
    # 动态调整X轴标签密度
    if len(filenames) > 30:
        step = len(filenames) // 30 + 1
        ax.set_xticks(x_indices[::step])
        ax.set_xticklabels(filenames[::step], rotation=45, ha='right', fontsize=9)
    else:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(filenames, rotation=45, ha='right', fontsize=9)

    ax.set_xlabel("模型训练进程 (文件列表顺序)", fontsize=12, fontweight='bold')

    # 4. 设置Y轴 (动态缩放以突显差异)
    y_min, y_max = min(scores), max(scores)
    y_range = y_max - y_min
    if y_range == 0: y_range = 1  # 防止除零
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.35)  # 顶部留更多空间给文字
    ax.set_ylabel("音高平均抖动量 (数值越低越好)", fontsize=12, fontweight='bold')

    # 5. 标题与标注
    ax.set_title("AI翻唱质量评估：音高稳定性 (Pitch Stability)", fontsize=16, pad=20)

    # 标记最优模型
    min_score_idx = np.argmin(scores)
    ax.annotate(f'最佳平滑度\n{filenames[min_score_idx]}',
                xy=(min_score_idx, scores[min_score_idx]),
                xytext=(min_score_idx, scores[min_score_idx] - y_range * 0.2),
                arrowprops=dict(facecolor='green', shrink=0.05),
                ha='center', color='green', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))

    # 6. 添加透明背景的文字描述
    text_desc = (
        "【指标解读】\n"
        "Y轴表示相邻两帧音高变化的平均幅度（音分）。\n"
        "• 曲线波动剧烈/数值高：代表声音有明显的锯齿感、机械电流音或基频断裂。\n"
        "• 曲线平缓/数值低：代表声音过渡自然，特别是长音和转音处理得当。\n"
        "• 观察趋势：寻找处于低位的 '山谷' 区域，通常对应最佳模型检查点。"
    )
    # 放置在图表左上角，透明背景
    ax.text(0.02, 0.98, text_desc, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.0, boxstyle='round,pad=0.5'))

    ax.legend(loc='upper right')
    plt.tight_layout()

    # --- 弹出窗口设置 ---

    # 设置窗口标题 (Window Title)
    try:
        fig.canvas.manager.set_window_title(f"分析结果 - 共{len(filenames)}个样本")
    except:
        pass

    # 尝试将窗口最大化 (仅限部分后端有效，如TkAgg)
    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager, 'window') and hasattr(manager.window, 'state'):
            manager.window.state('zoomed')  # Windows
        elif hasattr(manager, 'resize'):
            manager.resize(*manager.window.maxsize())  # Linux/Other
    except:
        pass  # 如果无法最大化，保持默认大小

    print(">>> 图表窗口已弹出，请在任务栏查看。关闭窗口后程序结束。")
    plt.show()  # 阻塞运行，直到手动关闭窗口


# --- 调用示例 ---
if __name__ == "__main__":
    # 请在此处填入你的wav文件路径列表
    # 示例：
    # my_files = ["/path/to/epoch_100.wav", "/path/to/epoch_200.wav", ...]
    # evaluate_pitch_stability_and_show(my_files)
    pass