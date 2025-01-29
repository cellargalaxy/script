import wave
import numpy as np
import matplotlib.pyplot as plt

# 读取 .wav 文件
file_path = '2025-01-29 15:50:59.wav'  # 修改为你自己的文件路径
wav_file = wave.open(file_path, 'r')

# 获取音频文件的基本信息
n_channels = wav_file.getnchannels()
sample_width = wav_file.getsampwidth()
frame_rate = wav_file.getframerate()
n_frames = wav_file.getnframes()

# 读取音频数据
frames = wav_file.readframes(n_frames)
audio_data = np.frombuffer(frames, dtype=np.int16)

# 如果是立体声，取单声道数据
if n_channels == 2:
    audio_data = audio_data[::2]  # 只取单声道的一个通道数据

# 创建时间轴
time = np.linspace(0, n_frames / frame_rate, num=n_frames)

# 绘制波形图
plt.figure(figsize=(10, 6))
plt.plot(time, audio_data)
plt.title("Waveform of the Audio File")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
