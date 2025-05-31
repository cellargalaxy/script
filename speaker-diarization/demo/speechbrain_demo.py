import torchaudio
from speechbrain.pretrained import SpeakerDiarization, EncoderDecoderASR
import os
from pydub import AudioSegment

# 设置路径
audio_file = "audio.wav"
tmp_dir = "tmp_speaker_diarization"
os.makedirs(tmp_dir, exist_ok=True)

# Step 1: 说话人分离
diarizer = SpeakerDiarization.from_hparams(
    source="speechbrain/diarization-ecapa-tdnn",
    savedir=os.path.join(tmp_dir, "diarization_model"),
    run_opts={"device": "cpu"}
)
diarizer.diarize_file(audio_file)

# Step 2: 加载 ASR 模型
asr = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-transformerlm-librispeech",
    savedir=os.path.join(tmp_dir, "asr_model"),
    run_opts={"device": "cpu"}
)

# Step 3: 读取 RTTM 结果
rttm_path = os.path.join(tmp_dir, "diarization_model", os.path.basename(audio_file).replace(".wav", ".rttm"))
segments = []
with open(rttm_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        start = float(parts[3])
        duration = float(parts[4])
        spk = parts[7]
        segments.append({"start": start, "end": start + duration, "speaker": spk})

# Step 4: 切割音频段并进行识别
audio = AudioSegment.from_wav(audio_file)
results = []

for i, seg in enumerate(segments):
    start_ms = int(seg["start"] * 1000)
    end_ms = int(seg["end"] * 1000)
    segment_audio = audio[start_ms:end_ms]
    segment_path = os.path.join(tmp_dir, f"segment_{i}.wav")
    segment_audio.export(segment_path, format="wav")

    # 加载并识别
    signal, fs = torchaudio.load(segment_path)
    text = asr.transcribe_batch(signal)

    results.append(f"[{seg['speaker']}] {text}")

# Step 5: 打印最终结果
print("\n--- Speaker-labeled Transcript ---\n")
for line in results:
    print(line)
