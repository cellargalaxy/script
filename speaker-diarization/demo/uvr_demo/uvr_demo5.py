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
    'UVR-DeNoise-Lite.pth',  #
    'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',  #
    'mel_band_roformer_denoise_debleed_gabox.ckpt',  #
    'mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt',  #
]
for model in models:
    separator = None
    try:
        separator = Separator(model_file_dir=os.path.join(util.get_home_dir(), '.cache', 'uvr'),
                              output_dir=os.path.join('output', model))
        separator.load_model(model_filename=model)
        output_files = separator.separate(['output/noreverb.wav'])
        if model == 'UVR-DeNoise-Lite.pth':
            util.delete_path('output/noreverb.wav')
            util.copy_file(os.path.join('output',model,'noreverb_(No Noise)_UVR-DeNoise-Lite.wav'), 'output/noreverb.wav')
        if model == 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt':
            util.delete_path('output/noreverb.wav')
            util.copy_file(os.path.join('output',model,'noreverb_(dry)_denoise_mel_band_roformer_aufr33_aggr_sdr_27.wav'), 'output/noreverb.wav')
        if model ==  'mel_band_roformer_denoise_debleed_gabox.ckpt':
            util.delete_path('output/noreverb.wav')
            util.copy_file(os.path.join('output',model,'noreverb_(Instrumental)_mel_band_roformer_denoise_debleed_gabox.wav'), 'output/noreverb.wav')
        if model ==   'mel_band_roformer_instrumental_fullness_noise_v4_gabox.ckpt':
            util.delete_path('output/noreverb.wav')
            util.copy_file(os.path.join('output',model,'noreverb_(Vocals)_mel_band_roformer_instrumental_fullness_noise_v4_gabox.wav'), 'output/noreverb.wav')
    except Exception as e:
        print(f'Failed to load model: {model}\nError: {e}')
    finally:
        print('Done')
