from faster_whisper import WhisperModel
from datetime import datetime
import os

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel("model/faster-whisper/medium", device="cpu", compute_type="int8")  # 至少得是medium，准确度才过得去

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

segments, info = model.transcribe(os.path.join(os.path.expanduser("~"), "download/27792247860-1-192.mp4"))

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
