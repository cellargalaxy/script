import torch
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import torch
import numpy as np

vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
    model_name="nvidia/frame_vad_multilingual_marblenet_v2.0", strict=False)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vad_model = vad_model.to(device)
vad_model.eval()

import librosa

# Load the audio
# input_signal = librosa.load("../2086-149220-0033.wav", sr=16000, mono=True)[0]
# input_signal = torch.tensor(input_signal).unsqueeze(0).float()
# input_signal_length = torch.tensor([input_signal.shape[1]]).long()

audio = AudioSegment.from_wav("../2086-149220-0033.wav")
samples = audio.get_array_of_samples()
input_signal = np.array(samples).astype(np.float32) / audio.max_possible_amplitude
input_signal = torch.tensor(input_signal).unsqueeze(0).float()
input_signal_length = torch.tensor([input_signal.shape[1]]).long()

# Perform inference
with torch.no_grad():
    torch_outputs = vad_model(
        input_signal=input_signal.to(device),
        input_signal_length=input_signal_length.to(device)
    ).cpu()
print("Original Logits (原始输出):")
print(torch_outputs)
print("-" * 30)

# --- 转换为概率 ---
# 在最后一个维度(dim=-1)上应用softmax函数
# 这个维度代表 [非语音, 语音] 两个类别
probs = torch.softmax(torch_outputs, dim=-1)

print("Probabilities (概率):")
print(probs)
print("-" * 30)

# --- 提取语音概率 ---
# 通常我们只关心“语音”的概率，它在第二个位置 (索引为 1)
speech_probs = probs[:, :, 1]

print("Speech Probability per Frame (每一帧是语音的概率):")
print(speech_probs)
print("-" * 30)

# --- (可选) 做出最终判决 ---
# 你可以设置一个阈值（比如0.5）来决定每一帧是否是语音
threshold = 0.5
speech_predictions = (speech_probs > threshold).int()

print(f"Final Decisions (with threshold={threshold}) (最终判决结果，1代表语音，0代表非语音):")
print(speech_predictions)