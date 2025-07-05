import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

from funasr import AutoModel

# 加载 ASR、VAD、标点和说话人识别模型
model = AutoModel(
    model="SenseVoiceSmall",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",  # 说话人聚类模型
    output_dir="outputs"  # 可选：保存临时切分片段与结果
)

# 推理
result = model.generate(input="../aaa/output/long/segment_divide/arrange.wav")

# 打印结果（包含每句话的说话人标签、文本、时间戳）
print("识别结果：\n")
for i, segment in enumerate(result["sentences"]):
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
          f"{segment['spk']}: {segment['text']}")
