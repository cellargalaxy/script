from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import json

model_dir ="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

# model = AutoModel(
#     model=model_dir,
#     vad_model="fsmn-vad",  # 语音活动检测
#     vad_kwargs={"max_single_segment_time": 30000},  # 表示VAD模型配置,max_single_segment_time: 表示vad_model最大切割音频时长, 单位是毫秒ms
#     punc_model="ct-punc",  # 标点模型
#     spk_model="cam++",  # 说话人分离模型
#     device="cpu",
#     mode="offline",  # 离线推理
# )

# res = model.generate(
#     input="short.wav",
#     cache={},
#     language="auto",
#     use_itn=True,  # 输出结果中是否包含标点与逆文本正则化
#     batch_size_s=60,  # 表示采用动态batch，batch中总音频时长，单位为秒s
#     merge_vad=True,  # 是否将 vad 模型切割的短音频碎片合成，合并后长度为merge_length_s，单位为秒s
#     merge_length_s=15,
# )

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
)

res = model.generate(
    input="short.wav",
    merge_length_s=15,
)
# text = rich_transcription_postprocess(res[0]["text"])
print('res', json.dumps(res))
