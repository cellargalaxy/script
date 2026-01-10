import os
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_sorted_wav_paths():
    """
    获取当前py文件所在文件夹里的全部wav文件路径，并按规则排序

    返回:
        list: 排序后的wav文件完整路径列表
    """
    # 获取当前py文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 获取目录中所有的.wav文件
    wav_files = []
    for file in os.listdir(current_dir):
        if file.lower().endswith('.wav'):
            full_path = os.path.join(current_dir, file)
            wav_files.append(full_path)

    # 定义排序函数
    def sort_key(file_path):
        # 获取文件名（不含路径）
        filename = os.path.basename(file_path)
        # 移除扩展名
        name_without_ext = os.path.splitext(filename)[0]

        # 检查文件名是否包含"_"
        if "_" in name_without_ext:
            # 使用"_"分割字符串
            parts = name_without_ext.split("_")
            # 检查最后一个部分是否是数字
            last_part = parts[-1]
            if last_part.isdigit():
                # 转换为整数用于排序
                return int(last_part)

        # 如果不满足条件，直接使用文件名进行排序
        return filename

    # 对文件进行排序
    sorted_files = sorted(wav_files, key=sort_key)

    return sorted_files


import multiprocessing as mp
from functools import partial


def import_and_run(module_name, func_name, wav_files):
    """在子进程中导入模块并执行函数"""
    try:
        module = __import__(module_name)
        func = getattr(module, func_name)
        return module_name, func(wav_files), None
    except Exception as e:
        return module_name, None, str(e)


def main():
    wav_files = get_sorted_wav_paths()

    # 定义所有要执行的分析任务
    tasks = [
        ("集成响度", "analyze_integrated_loudness"),
        ("短时响度波动", "analyze_short_term_loudness_variance"),
        ("音高与基频稳定性", "analyze_f0_stability"),
        ("音高漂移", "analyze_pitch_drift"),
        ("峰均比", "analyze_crest_factor"),
        ("动态范围", "analyze_rms_dynamic_range"),
        ("削波检测", "analyze_clipping_detection"),
        ("频谱质心", "analyze_spectral_centroid"),
        ("频谱平坦度", "analyze_spectral_flatness"),
        ("谐噪比", "analyze_hnr_quality"),
        ("频谱通量", "analyze_spectral_flux"),
        ("频谱熵", "analyze_spectral_entropy"),
        ("频谱带宽", "evaluate_spectral_bandwidth"),
        ("频谱滚降点", "analyze_spectral_rolloff"),
        ("共振峰分析", "analyze_formant_quality"),
        ("振幅微扰", "analyze_shimmer_quality"),
        ("零交叉率", "analyze_zcr_quality"),
        ("高频能量异常", "analyze_hf_energy_ratio"),
        ("信噪比估算", "analyze_audio_snr"),
        ("总谐波失真", "analyze_thdn"),
        ("频谱空洞与高频缺失", "analyze_high_frequency_quality"),
        ("MFCC距离分析", "analyze_mfcc_distance"),
        ("调制频谱分析", "analyze_modulation_spectrum"),
        ("连续性指标", "analyze_frame_continuity"),
    ]

    # 创建进程池
    cpu_count = mp.cpu_count()
    pool_size = min(len(tasks), cpu_count * 2)  # 合理的进程数

    print(f"开始并行分析，使用 {pool_size} 个进程...")

    with mp.Pool(processes=pool_size) as pool:
        # 使用partial固定wav_files参数
        run_task = partial(import_and_run, wav_files=wav_files)

        # 异步执行所有任务
        results = []
        for module_name, func_name in tasks:
            result = pool.apply_async(run_task, (module_name, func_name))
            results.append((module_name, result))

        # 收集结果
        completed_results = {}
        errors = {}

        for module_name, result in results:
            try:
                name, data, error = result.get(timeout=300)  # 5分钟超时
                if error:
                    errors[name] = error
                    print(f"❌ {name}: 失败 - {error}")
                else:
                    completed_results[name] = data
                    print(f"✅ {name}: 完成")
            except mp.TimeoutError:
                errors[module_name] = "超时"
                print(f"❌ {module_name}: 超时")
            except Exception as e:
                errors[module_name] = str(e)
                print(f"❌ {module_name}: 异常 - {e}")

    print(f"\n分析完成!")
    print(f"成功: {len(completed_results)} 个")
    print(f"失败: {len(errors)} 个")

    return completed_results, errors


if __name__ == "__main__":
    # 在Windows上使用多进程必须要有这个判断
    mp.freeze_support()
    results, errors = main()