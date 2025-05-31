import whisperx

# Step 1: 加载模型（默认是 whisper-large，建议使用 tiny 或 base 模型以节省资源）
model = whisperx.load_model("base", device="cpu")

# Step 2: 运行转录
audio = "208253969-7e35fe2a-7541-434a-ae91-8e919540555d.wav"
transcription = model.transcribe(audio)

print(transcription["segments"])  # 这会包含每句话的时间戳

# Step 3: 对齐单词
align_model, metadata = whisperx.load_align_model(language_code="en", device="cpu")
transcription_aligned = whisperx.align(transcription["segments"], align_model, metadata, audio, device="cpu")

from huggingface_hub import login
login(token="")

# 加载说话人分离模型（可选：传入 num_speakers 限定人数）
diarize_model = whisperx.DiarizationPipeline(use_auth_token="", device="cpu")

# 执行说话人分离（需要原始音频路径）
diarize_segments = diarize_model(audio)

# 将分离信息合并到 transcription 中
result = whisperx.assign_speakers(transcription_aligned, diarize_segments)

# 查看结果
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] Speaker {segment['speaker']}: {segment['text']}")
