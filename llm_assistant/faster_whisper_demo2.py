from faster_whisper import WhisperModel
from pydub import AudioSegment

# 加载 Whisper 模型
model = WhisperModel("model/faster-whisper/base", device="cpu", compute_type="int8")

# 加载音频文件
audio_path = "2025-01-31 17:36:32-wav.wav"
audio = AudioSegment.from_file(audio_path)

# 对音频进行转录
segments, _ = model.transcribe(audio_path)

# 获取每个句子的开始和结束时间
sentence_boundaries = [(segment.start, segment.end) for segment in segments]

# 根据句子的时间边界切割音频
for idx, (start, end) in enumerate(sentence_boundaries):
    # 转换时间为毫秒
    start_ms = start * 1000
    end_ms = end * 1000

    # 切割音频
    sentence_audio = audio[start_ms:end_ms]

    # 保存每个句子的音频文件
    sentence_audio.export(f"sentence_{idx + 1}.mp3", format="mp3")
    print(f"保存句子 {idx + 1} 的音频文件: sentence_{idx + 1}.mp3")