"""
写一个python脚本，只用标准库，无外部依赖
环境变量中有ffmpeg与ffprobe
1. 脚本启动后，手动在终端输入多个参数
1.1 输入文件夹路径，必填，例如d:/mp3
1.2 输出文件夹路径，选填，不填默认为，输入文件夹所在的文件夹的下的wav文件夹，例如d:/wav，不存在则创建
1.3 选择采样率，可选，选择类型有16kHz/44.1kHz/48kHz，如果不填则默认原文件的采样率
2. 遍历输入文件夹下的全部文件，转为wav文件，以相同文件名称，保存在输出文件夹下
2.1 尽可能的保留源文件的声道/数据深度/采样率（除非在入参时手动选择了）等信息，尽可能保留原文件的数据质量
2.2 在终端输出处理日志
3. 最后点击任意键退出脚本
"""

import os
import subprocess


def input_with_default(prompt, default=None):
    user_input = input(f"{prompt} [{default if default is not None else ''}]: ")
    if not user_input.strip() and default is not None:
        return default
    return user_input.strip()


def get_audio_info(filepath):
    """Return dict with sample_rate, channels, bit_depth from ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate,channels,bit_depth',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        info = {}
        # ffprobe sometimes doesn't provide bit_depth, so handle gracefully
        if len(lines) >= 2:
            info['sample_rate'] = lines[0]
            info['channels'] = lines[1]
            info['bit_depth'] = lines[2] if len(lines) > 2 else None
        return info
    except Exception as e:
        print(f"Failed to get info for {filepath}: {e}")
        return {}


def safe_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def main():
    print("=== 音频批量转 WAV 工具 ===")
    # 1. 输入参数
    in_dir = ''
    while not in_dir or not os.path.isdir(in_dir):
        in_dir = input("请输入【输入文件夹路径】（必填，例如 d:/mp3）：").strip()
        if not os.path.isdir(in_dir):
            print("路径不存在或不是文件夹，请重新输入。")

    default_out_dir = os.path.join(os.path.dirname(in_dir.rstrip('/\\')), 'wav')
    out_dir = input_with_default("请输入【输出文件夹路径】（选填，默认同级 wav 文件夹，例如 d:/wav）", default_out_dir)
    safe_makedirs(out_dir)

    # 采样率选择
    sample_rates = {
        '1': '16000',
        '2': '44100',
        '3': '48000'
    }
    print("可选采样率：1) 16kHz  2) 44.1kHz  3) 48kHz")
    sr_choice = input_with_default("请选择采样率（输入序号，默认原文件采样率）", "")
    sample_rate = sample_rates.get(sr_choice, None)

    # 2. 遍历文件
    print(f"\n开始遍历文件夹 {in_dir} ...")
    files = []
    for root, _, filenames in os.walk(in_dir):
        for fname in filenames:
            files.append(os.path.join(root, fname))

    if not files:
        print("输入文件夹下没有文件。")
    else:
        for idx, file_path in enumerate(files, 1):
            file_name = os.path.basename(file_path)
            out_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + ".wav")
            print(f"[{idx}/{len(files)}] 正在处理: {file_name}")

            info = get_audio_info(file_path)
            # 构造 ffmpeg 命令
            ffmpeg_cmd = ['ffmpeg', '-y', '-i', file_path]

            # 保持声道数
            if 'channels' in info and info['channels']:
                ffmpeg_cmd += ['-ac', info['channels']]

            # 保持数据深度
            # ffmpeg 默认会用原位深度，除非强制指定
            # WAV 格式通常支持 16/24/32 位，若 info['bit_depth'] 存在可指定如下
            if 'bit_depth' in info and info['bit_depth']:
                ffmpeg_cmd += ['-sample_fmt', f's{info["bit_depth"]}']

            # 采样率
            if sample_rate:
                ffmpeg_cmd += ['-ar', sample_rate]
            elif 'sample_rate' in info and info['sample_rate']:
                ffmpeg_cmd += ['-ar', info['sample_rate']]

            ffmpeg_cmd.append(out_path)

            try:
                proc = subprocess.run(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # 统一用 utf-8 解码，避免 UnicodeDecodeError
                stderr_text = proc.stderr.decode('utf-8', errors='replace')
                if proc.returncode == 0:
                    print(f"    转换成功: {out_path}")
                else:
                    print(
                        f"    转换失败: {file_name}\n    错误: {stderr_text.splitlines()[-1] if stderr_text else '未知'}")
            except Exception as e:
                print(f"    ffmpeg 调用异常: {e}")

    print("\n全部处理完成。按任意键退出。")
    input()


if __name__ == "__main__":
    main()
