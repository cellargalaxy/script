import nemo
import nemo.collections.asr as nemo_asr
import torchaudio

# 加载 NeMo 的预训练 Conformer 模型（英文）
asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained("stt_en_conformer_ctc_large")

# 输入你的音频路径（支持 .wav/.mp3）
audio_path = "short.wav"

# 进行转录（字幕识别）
transcript = asr_model.transcribe([audio_path])
print("Transcript:", transcript[0])
