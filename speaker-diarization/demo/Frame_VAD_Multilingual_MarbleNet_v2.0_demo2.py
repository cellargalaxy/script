import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

# 不适合，识别不出某些人声，可能与他的训练策略有关
# 他的文档描述：为了减少假阳性错误（即模型错误地将没有语音的声音识别为语音的情况），该模型使用了白噪声和真实噪声扰动进行训练

import torch
import nemo.collections.asr as nemo_asr
import math
import sub_util
import util
import librosa
import torch.nn.functional as F
import matplotlib.pyplot as plt


def plot_vad_segments(segments):
    """
    绘制VAD段落概率图

    参数:
        segments: 列表，每个元素是字典，包含 'start'（毫秒）, 'end'（毫秒）, 'vad_probability'（0~1）
    """
    if not segments:
        print("segments 为空")
        return

    # 构建阶梯线
    xs = []
    ys = []
    for seg in segments:
        xs.extend([seg["start"], seg["end"]])
        ys.extend([seg["vad_probability"], seg["vad_probability"]])

    plt.figure(figsize=(10, 4))
    plt.step(xs, ys, where='post', label="VAD Probability")
    plt.xlabel("Time (ms)")
    plt.ylabel("VAD Probability")
    plt.ylim(0, 1.05)
    plt.title("Voice Activity Detection Segments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def vad(audio_path):
    device = util.get_device_type()

    vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
        model_name="nvidia/frame_vad_multilingual_marblenet_v2.0")
    vad_model = vad_model.to(device)
    vad_model.eval()

    input_signal = librosa.load(audio_path, sr=16000, mono=True)[0]
    input_signal = torch.tensor(input_signal).unsqueeze(0).float()
    input_signal_length = torch.tensor([input_signal.shape[1]]).long()
    with torch.no_grad():
        torch_outputs = vad_model(
            input_signal=input_signal.to(device),
            input_signal_length=input_signal_length.to(device),
        ).cpu()

    logits = torch_outputs[0]
    probs_all = F.softmax(logits, dim=-1)
    speech_probs = probs_all[:, 1].cpu().numpy()

    segments = []
    for i, probability in enumerate(speech_probs):
        vad_type = 'silene'
        if probability >= 0.2:
            vad_type = 'speech'
        pre_end = 0
        if len(segments) > 0:
            pre_end = segments[len(segments) - 1]['end']
        end = math.floor((i + 1) * 0.02 * 1000)
        if len(segments) == 0:
            segments.append({"start": pre_end, "end": end, "vad_type": vad_type, "vad_probabilitys": [probability]})
            continue
        if segments[len(segments) - 1]['vad_type'] == vad_type:
            segments[len(segments) - 1]['end'] = end
            segments[len(segments) - 1]['vad_probabilitys'].append(probability)
            continue
        segments.append({"start": pre_end, "end": end, "vad_type": vad_type, "vad_probabilitys": [probability]})
    for i, segment in enumerate(segments):
        segments[i]['vad_probability'] = sum(segments[i]['vad_probabilitys']) / len(segments[i]['vad_probabilitys'])
    for i, segment in enumerate(segments):
        if 0.8 <= segments[i]['vad_probability']:
            segments[i]['vad_type'] = 'speech'
            continue
        if segments[i]['vad_probability'] <= 0.5:
            segments[i]['vad_type'] = 'silene'
            continue
        if segments[i]['vad_type'] == 'speech' and segments[i]['end'] - segments[i]['start'] < 250:
            segments[i]['vad_type'] = 'silene'
    return segments


segments = vad('../demo_eng_single.wav')
sub_util.check_segments(segments)
sub_util.save_segments_as_srt(segments, '../demo_eng_single.srt', skip_silene=True)
plot_vad_segments(segments)
