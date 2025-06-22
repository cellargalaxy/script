from pyannote.audio import Pipeline
import math
import sub_util
import  util
import torch

# ========== 配置 ==========
AUDIO_FILE = "../demo_jpn_single.wav"
HUGGINGFACE_TOKEN = ""  # 替换为你自己的 token
MIN_SILENCE_DURATION = 0.5  # 最小静音长度（秒），避免在太短静音处切割
USE_SEGMENT_MIDPOINT = True  # 是否使用静音段中点作为切点

pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HUGGINGFACE_TOKEN)
pipeline = pipeline.to(torch.device(util.get_device_type()))
vad_result = pipeline(AUDIO_FILE)

segments = []
for speech_segment in vad_result.get_timeline():
    print(f"start={speech_segment.start}s stop={speech_segment.end}s")
    start = math.floor(speech_segment.start * 1000)
    end = math.floor(speech_segment.end * 1000)
    segments.append({"start": start, "end": end, "vad_type": 'speech'})

sub_util.save_segments_as_srt(segments, '../demo_jpn_single.srt')
