import os
import glob
import 集成响度


def get_sorted_wav_paths(folder_path=None):
    """
    获取文件夹中所有wav文件的路径，并按升序排序

    参数:
    folder_path: 文件夹路径，默认为当前文件所在文件夹

    返回:
    list: 排序后的wav文件路径列表
    """
    # 如果未指定文件夹路径，使用当前文件所在文件夹
    if folder_path is None:
        folder_path = os.path.dirname(os.path.abspath(__file__))

    # 使用glob查找所有wav文件
    wav_files = glob.glob(os.path.join(folder_path, "*.wav"))

    # 按文件名升序排序
    wav_files.sort()

    return wav_files


wav_files = get_sorted_wav_paths()
集成响度.analyze_integrated_loudness(wav_files)
