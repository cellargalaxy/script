from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import torch

# 替换为你自己的 Hugging Face Token
HUGGINGFACE_TOKEN = ""
AUDIO_FILE = "../gen_sub/output/demo/noise_reduction/htdemucs/wav/vocals.wav"

# 加载 VAD pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=HUGGINGFACE_TOKEN)

# 推理得到语音活动（segments）
vad_result = pipeline(AUDIO_FILE)

# 使用私有API获得每帧的置信度（0~1）
inference = pipeline._models
output = inference({'audio': AUDIO_FILE})

# 获取时间轴和分数
time = output['frames'].numpy()
score = output['scores'].numpy()

# 画图显示置信度
plt.figure(figsize=(12, 4))
plt.plot(time, score, label='Speech Probability')
plt.xlabel("Time (s)")
plt.ylabel("Probability")
plt.title("VAD Confidence Scores Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 从预测结果中提取“静音”时间段（即不在语音段内）
print("Silent Segments:")
full_audio = vad_result.get_timeline().extent()
speech = vad_result.get_timeline()
silence = speech.support().invert().crop(full_audio, mode='intersection')
for segment in silence:
    print(f"Start: {segment.start:.2f}s, End: {segment.end:.2f}s")
