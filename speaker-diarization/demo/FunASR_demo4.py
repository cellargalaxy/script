from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import json

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
    model="Whisper-large-v3", model_revision="v2.0.4",
    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
    spk_model="cam++", spk_model_revision="v2.0.2",
    device="cpu",
    mode="offline",
    batch_size=1,
)
res = model.generate(
    input="short.wav",
    merge_length_s=15,
    batch_size=1,
    batch_size_s=5,
)
print('res', json.dumps(res))
