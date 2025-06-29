import os

os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

config = OmegaConf.load("nemo_config.yaml")
diar_model = ClusteringDiarizer(cfg=config)

diar_model.diarize()
