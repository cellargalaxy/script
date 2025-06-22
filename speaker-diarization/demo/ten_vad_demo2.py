from ten_vad import TenVad
import scipy.io.wavfile as Wavfile
import math
import sub_util
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

def vad(audio_path, frame_rate=10):
    simple_rate, data = Wavfile.read(audio_path)
    frame_simple = int(simple_rate / frame_rate)  # 每侦采样数
    frame_cnt = data.shape[0] // frame_simple  # 总帧数
    ten_vad_instance = TenVad(frame_simple)
    segments = []
    for i in range(frame_cnt):
        audio_data = data[i * frame_simple: (i + 1) * frame_simple]
        probability, _ = ten_vad_instance.process(audio_data)
        vad_type = 'silene'
        if probability >= 0.4:
            vad_type = 'speech'
        pre_end = 0
        if len(segments) > 0:
            pre_end = segments[len(segments) - 1]['end']
        end = math.floor((i + 1) * (1.0 / frame_rate) * 1000)
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



def vad_plt(audio_path, frame_rate=10, threshold=0.9):
    simple_rate, data = Wavfile.read(audio_path)
    frame_simple = int(simple_rate / frame_rate)  # 每帧采样数
    frame_cnt = data.shape[0] // frame_simple  # 总帧数
    ten_vad_instance = TenVad(frame_simple, threshold)
    probabilities = []
    flags = []
    times = []
    for i in range(frame_cnt):
        audio_data = data[i * frame_simple: (i + 1) * frame_simple]
        probability, flag = ten_vad_instance.process(audio_data)
        probabilities.append(probability)
        flags.append(flag)
        times.append((i + 0.5) * frame_simple / simple_rate)
    plt.figure(figsize=(12, 4))
    plt.plot(times, probabilities, label="VAD Probability")
    plt.xlabel("Time (s)")
    plt.ylabel("Speech Probability")
    plt.title("Framewise Speech Probability by VAD")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# vad_plt('../demo_eng_single.wav')
