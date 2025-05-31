import whisperx
import ffmpeg
import os
import torch

# 输入音频路径
audio_path = "208253969-7e35fe2a-7541-434a-ae91-8e919540555d.mp4"
output_dir = "speaker_segments"
os.makedirs(output_dir, exist_ok=True)
compute_type = "int8"  # change to "int8" if low on GPU mem (may reduce accuracy) float16

# Step 1: Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# Step 2: Transcribe with word-level timestamps
transcription = model.transcribe(audio_path, batch_size=16)

# Step 3: Diarization (speaker labeling)
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="", device=device)
diarize_segments = diarize_model(audio_path)

# Step 4: Align speaker labels with transcription
aligned_segments = whisperx.align(transcription["segments"], diarize_segments)

# Step 5: Cut audio per speaker segment
for i, seg in enumerate(aligned_segments):
    start = seg["start"]
    end = seg["end"]
    speaker = seg["speaker"]
    text = seg["text"].strip().replace(" ", "_")[:30]

    output_path = os.path.join(output_dir, f"{i:03d}_{speaker}_{text}.wav")

    (
        ffmpeg
        .input(audio_path, ss=start, to=end)
        .output(output_path, acodec="pcm_s16le", ac=1, ar="16k")
        .overwrite_output()
        .run(quiet=True)
    )

print("音频切割完成，每个说话人的片段已保存到:", output_dir)
