from pydub import AudioSegment
import numpy as np

# 加载音频文件
audio = AudioSegment.from_file("mywav_reduced_noise.wav")

# 将音频转为 numpy 数组
samples = np.array(audio.get_array_of_samples())

# 基本的噪声减少方法（你可以根据需要调节）
# 例如: 用一个简单的去噪方法来平滑一些突发的爆鸣声
# 这里我们做一个简单的平滑处理，去掉极值

samples = np.clip(samples, -10000, 10000)

# 转换回音频格式
audio = audio._spawn(samples.tobytes())

# 保存处理后的音频
audio.export("processed_audio.wav", format="wav")
