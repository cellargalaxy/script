from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import json

model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                  spk_model="cam++", spk_model_revision="v2.0.2",
                  )
input = 'short.wav'
res = model.generate(input=input,
            batch_size_s=300,
            hotword='魔搭')
print('res', json.dumps(res))
