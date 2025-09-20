import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import util
import os
from audio_separator.separator import Separator

# 降噪
models = [
    # 'UVR-DeNoise-Lite.pth',  # 沙哑有好多
    # 'UVR-DeNoise.pth',  # 沙哑有去掉，但是不少有漏
    # 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',  # 沙哑有去掉，但是不少有漏
    # 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt', # 沙哑有去掉，但是不少有漏
    # 'mel_band_roformer_denoise_debleed_gabox.ckpt',  # 声音都全压丢了
    # 'mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt',

    # 效果都不好
    # 没有一个是能把RipX里识别出的那种噪声较好去掉的，甚至会加上更多
]
for model in models:
    separator = None
    try:
        separator = Separator(model_file_dir=os.path.join(util.get_home_dir(), '.cache', 'uvr'),
                              output_dir=os.path.join('output', model))
        separator.load_model(model_filename=model)
        output_files = separator.separate(['../../material/noreverb.wav'])
    except Exception as e:
        print(f'Failed to load model: {model}\nError: {e}')
    finally:
        print('Done')
