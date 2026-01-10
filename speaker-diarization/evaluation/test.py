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


wav_files = get_sorted_wav_paths()
# import 集成响度;集成响度.analyze_integrated_loudness(wav_files)
# import 短时响度波动;短时响度波动.analyze_short_term_loudness_variance(wav_files)
# import 音高与基频稳定性;音高与基频稳定性.analyze_f0_stability(wav_files)
# import 音高漂移;音高漂移.analyze_pitch_drift(wav_files)
# import 峰均比;峰均比.analyze_crest_factor(wav_files)
# import 动态范围;动态范围.analyze_rms_dynamic_range(wav_files)
# import 削波检测;削波检测.analyze_clipping_detection(wav_files)
# import 频谱质心;频谱质心.analyze_spectral_centroid(wav_files)
# import 频谱平坦度;频谱平坦度.analyze_spectral_flatness(wav_files)
# import 谐噪比;谐噪比.analyze_hnr_quality(wav_files)
import 频谱通量;频谱通量.analyze_spectral_flux(wav_files)
