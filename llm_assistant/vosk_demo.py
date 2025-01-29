import pyaudio
import vosk
import json
import time

# 初始化Vosk模型
model = vosk.Model("./model/vosk/vosk-model-cn-0.22") #vosk-model-cn-0.22的准确率比small高不少，并且内存和CPU好像没有多太多
recognizer = vosk.KaldiRecognizer(model, 16000)

# 打开音频流
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
stream.start_stream()

print("开始语音识别...")

while True:
    data = stream.read(4000)
    if recognizer.AcceptWaveform(data):
        result = recognizer.Result()
        result_dict = json.loads(result)
        print("识别结果:", result_dict['text'])
    else:
        partial_result = recognizer.PartialResult()
        partial_dict = json.loads(partial_result)
        print("部分识别:", partial_dict['partial'])

    time.sleep(0.1)
