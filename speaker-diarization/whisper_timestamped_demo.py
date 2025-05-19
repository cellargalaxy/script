import whisper_timestamped as whisper

audio = whisper.load_audio("208253969-7e35fe2a-7541-434a-ae91-8e919540555d.wav")

model = whisper.load_model("large-v3", device="cpu")

result = whisper.transcribe(model, audio, language="en")

import json
print(json.dumps(result, indent = 2, ensure_ascii = False))