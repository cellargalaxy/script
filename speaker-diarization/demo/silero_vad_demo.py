import torch
import pysubs2


def save_segments_as_srt(segments, file_path):
    results = []
    for i, segment in enumerate(segments):
        start = segment['start']
        end = segment['end']
        text = segment.get('text', '')
        if not text:
            text = f"[{segment.get('type', '')}|{segment.get('speaker', '')}] {segment['start']}->{segment['end']}"
        obj = {'start': start, 'end': end, 'text': text}
        results.append(obj)
    subs = pysubs2.load_from_whisper(results)
    subs.save(file_path)


# 加载模型（CPU/GPU 均可）
model, utils = torch.hub.load(repo_or_dir='../model/silero-vad/master', model='silero_vad', source='local')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# 读取音频文件（必须是 WAV 格式，16kHz 单声道）
wav_path = '../demo_eng.wav'  # 替换为你的音频路径
wave = read_audio(wav_path, sampling_rate=16000)

# 获取语音片段（语音活动检测结果）
speech_timestamps = get_speech_timestamps(
    wave, model,
    sampling_rate=16000,
    min_silence_duration_ms=10,  # 最短要多长的静音（ms）才被认为是真正的停顿
    min_speech_duration_ms=1000,  # 设置为更短可以识别短句
    threshold=0.6,# 默认是 0.5，越小越敏感
)

# 打印检测结果
segments = []
for i, segment in enumerate(speech_timestamps):
    start_sec = segment['start'] / 16000
    end_sec = segment['end'] / 16000
    print(f"Speech {i + 1}: {start_sec:.2f}s - {end_sec:.2f}s")
    segments.append({"start": start_sec, "end": end_sec, })
save_segments_as_srt(segments, 'silero_vad_demo.srt')
