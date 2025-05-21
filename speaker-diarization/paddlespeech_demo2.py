from speechbrain.inference.separation import SepformerSeparation as separator
import json

model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')
est_sources = model.separate_file(path='short.wav')
print(est_sources)
# print('est_sources', json.dumps(est_sources))