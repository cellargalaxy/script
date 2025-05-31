# key

from pyannote.audio import Pipeline
from pyannote.core import Segment
import wave

# ========== 配置 ==========
AUDIO_FILE = "../short.wav"
HUGGINGFACE_TOKEN = ""  # 替换为你自己的 token
MIN_SILENCE_DURATION = 0.5  # 最小静音长度（秒），避免在太短静音处切割
USE_SEGMENT_MIDPOINT = True  # 是否使用静音段中点作为切点


# ========== 获取音频总时长 ==========
def get_audio_duration(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


duration = get_audio_duration(AUDIO_FILE)

# ========== 运行 pyannote VAD ==========
pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=HUGGINGFACE_TOKEN
)

vad_result = pipeline(AUDIO_FILE)

# ========== 推导静音段 ==========
non_speech_segments = []
previous_end = 0.0

for speech_segment in vad_result.get_timeline():
    if speech_segment.start > previous_end:
        non_speech_segments.append(Segment(previous_end, speech_segment.start))
    previous_end = speech_segment.end

if previous_end < duration:
    non_speech_segments.append(Segment(previous_end, duration))

# ========== 提取可切割点 ==========
cut_points = []

for seg in non_speech_segments:
    silence_duration = seg.end - seg.start
    if silence_duration >= MIN_SILENCE_DURATION:
        cut_time = (seg.start + seg.end) / 2 if USE_SEGMENT_MIDPOINT else seg.start
        cut_points.append(round(cut_time, 2))

# ========== 输出结果 ==========
print("建议切割点（单位：秒）:")
for idx, t in enumerate(cut_points):
    print(f"{idx + 1}. {t:.2f} s")
