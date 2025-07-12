import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

import util
import os
from audio_separator.separator import Separator

models = [
    "vocals_mel_band_roformer.ckpt",
    "melband_roformer_big_beta4.ckpt",#有些纵深
    "mel_band_roformer_kim_ft_unwa.ckpt",
    "melband_roformer_big_beta5e.ckpt",#有些纵深
    "MelBandRoformerBigSYHFTV1.ckpt",#有些纵深
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "Kim_Vocal_2.onnx",#有些干，但只有一次有影子
]
for model in models:
    separator = None
    try:
        separator = Separator(model_file_dir=os.path.join(util.get_home_dir(), '.cache', 'uvr'),
                              output_dir=os.path.join('output', model))
        separator.load_model(model_filename=model)
        output_files = separator.separate(['../../long_jpn.wav'])
    except Exception as e:
        print(f'Failed to load model: {model}\nError: {e}')
    finally:
        print('Done')
