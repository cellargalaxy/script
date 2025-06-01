import torch
import torchaudio
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice

# 设置 CPU 模式
device = torch.device('cpu')

# 初始化 TTS 模型（不使用 DeepSpeed，兼容 CPU）
tts = TextToSpeech(use_deepspeed=False, device=device)

# 加载自定义语音样本文件夹 voices/my_voice/
voice_samples, conditioning_latents = load_voice("my_voice")

# 要合成的文字（可以换成你自己的句子）
text = "The wolf was trustworthy and struck a promise with the village youth."

# 使用 fast 模式合成（标准、high_quality 会非常慢）
generated_audio = tts.tts_with_preset(
    text=text,
    voice_samples=voice_samples,
    conditioning_latents=conditioning_latents,
    preset='fast'  # 可选: ultra_fast / fast / standard / high_quality
)

# 保存音频
torchaudio.save("output.wav", generated_audio.squeeze(0).cpu(), sample_rate=24000)

print("✅ 合成完成，语音保存为 output.wav")
