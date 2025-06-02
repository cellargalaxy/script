from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import torch
import torchaudio

# 指向包含音频样本的文件夹路径
print('指向包含音频样本的文件夹路径')
voice_sample_folder = "../aaa/output/demo/audio_class"  # 替换为你的文件夹路径

# 加载音色
print('加载音色')
voice_samples, conditioning_latents = load_voice('SPEAKER_02', extra_voice_dirs=[voice_sample_folder])

# 需要合成的文本
print('需要合成的文本')
text = "The wolf was trustworthy and struck a promise with the village youth."

# 初始化TTS
print('初始化TTS')
device = torch.device('cpu')
tts = TextToSpeech(use_deepspeed=False, device=device)

# 合成语音
print('合成语音')
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
              preset='fast',# 可选: ultra_fast / fast / standard / high_quality
              # num_autoregressive_samples=1,
              # diffusion_iterations=10,
              # cond_free=True
              )

# 保存生成的音频
print('保存生成的音频')
torchaudio.save("output.wav", gen.squeeze(0).cpu(), sample_rate=24000)
