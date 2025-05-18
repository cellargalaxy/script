# 这个可用
# 模型文件缓存
# cd ~/.cache/torch/pyannote
# ll
# 总计 0
# drwxr-xr-x 1 user user 36  5月18日 11:54 models--pyannote--segmentation-3.0
# drwxr-xr-x 1 user user 36  5月18日 11:54 models--pyannote--speaker-diarization-3.1
# drwxr-xr-x 1 user user 36  5月18日 11:54 models--pyannote--wespeaker-voxceleb-resnet34-LM
#
# cp -r pyannote .


import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cpu"))
diarization = pipeline("input.wav")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
