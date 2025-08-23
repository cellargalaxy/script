import os
import struct
import glob

def wav_duration(path: str) -> float:
    """通过文件大小和 WAV header 计算音频时长（秒，仅支持 PCM/未压缩）"""
    with open(path, "rb") as f:
        header = f.read(44)
        if header[0:4] != b'RIFF' or header[8:12] != b'WAVE':
            raise ValueError(f"{path} 不是标准 WAV 文件")

        num_channels = struct.unpack("<H", header[22:24])[0]
        sample_rate = struct.unpack("<I", header[24:28])[0]
        bits_per_sample = struct.unpack("<H", header[34:36])[0]

    file_size = os.path.getsize(path)
    header_size = 44
    data_size = file_size - header_size
    byte_rate = sample_rate * num_channels * bits_per_sample // 8

    return data_size / byte_rate


if __name__ == "__main__":
    wav_files = glob.glob("*.wav")
    if not wav_files:
        print("当前文件夹没有找到 wav 文件")
        exit(0)

    total_seconds = 0.0
    for wav in wav_files:
        try:
            dur = wav_duration(wav)
            print(f"{wav} -> {dur:.2f} 秒")
            total_seconds += dur
        except Exception as e:
            print(f"{wav} 错误: {e}")

    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    print("\n====== 汇总结果 ======")
    print(f"总时长: {total_seconds:.2f} 秒")
    print(f"约 {total_minutes:.2f} 分钟")
    print(f"约 {total_hours:.2f} 小时")

