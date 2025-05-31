from pyannote.audio import Pipeline
import soundfile as sf

def trim_silence_pyannote(input_path, output_path):
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=None) # 去除 静音
    vad_result = pipeline(input_path)

    if not vad_result:
        print("未检测到语音")
        return

    # 获取语音区域的起止时间
    speech_segments = list(vad_result.itersegments())
    start_time = speech_segments[0].start
    end_time = speech_segments[-1].end

    # 加载原始音频
    audio, sr = sf.read(input_path)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    trimmed_audio = audio[start_sample:end_sample]
    sf.write(output_path, trimmed_audio, sr)
    print(f"输出文件已保存: {output_path}")

# 示例用法
trim_silence_pyannote("../short.wav", "trim.wav")
