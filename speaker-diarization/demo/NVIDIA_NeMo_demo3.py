import os

os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

import json

meta = {
    'audio_filepath': '../demo_eng_single.wav',
    'offset': 0,
    'duration': None,
    'label': 'infer',
    'text': '-',
    'num_speakers': None,
    'rttm_filepath': None,
    'uem_filepath': None
}

with open('input_manifest.json', 'w') as fp:
    json.dump(meta, fp)
    fp.write('\n')

config = OmegaConf.load("nemo_config.yaml")
config.diarizer.manifest_filepath = 'input_manifest.json'
config.diarizer.out_dir = 'nemo_output'
model = ClusteringDiarizer(cfg=config)
model.diarize()
