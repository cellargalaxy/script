from types import SimpleNamespace
from denoiser import pretrained
from denoiser.enhance import enhance

# 构造 args 参数
args = SimpleNamespace(
    out_dir="output",         # 输出目录
    device="cpu",             # 或 "cuda"
    noisy_dir="aaaaa/output/demo/extract_audio_track/wav.wav",        # 输入音频路径
)

# 加载模型
model = pretrained.dns64()

# 调用增强函数
enhance(args, model=model)
