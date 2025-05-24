from pyannote.audio import Pipeline
from pyannote.core import Segment
import wave

# ========== 配置 ==========
AUDIO_FILE = "short.wav"
HUGGINGFACE_TOKEN = ""  # 替换为你自己的 token
MIN_SILENCE_DURATION = 0.5  # 最小静音长度（秒），避免在太短静音处切割
USE_SEGMENT_MIDPOINT = True  # 是否使用静音段中点作为切点

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HUGGINGFACE_TOKEN)
vad_result = pipeline(AUDIO_FILE)

for speech_segment in vad_result.get_timeline():
    print(f"start={speech_segment.start}s stop={speech_segment.end}s")


