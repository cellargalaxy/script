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

# 去混音
models = [
    # "Reverb_HQ_By_FoxJoy.onnx",  # 有效果，但是有漏，比MDX23C漏                        64M
    # "MDX23C-De-Reverb-aufr33-jarredou.ckpt",  # 有效果，但是有漏                        428M
    # "dereverb_super_big_mbr_ep_346.ckpt",    # 有效果，但是有漏                            435M
    # "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",  # 有效果，有少少漏                  798M
    # "dereverb_big_mbr_ep_362.ckpt",   # 好，好像有少少漏                                     435M
    # "UVR-DeEcho-DeReverb.pth", # 有效果，但是有漏                                        214M
    # "UVR-De-Reverb-aufr33-jarredou.pth",   # 明显有漏                                57M

    # "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",  # 好，但有些奇怪的破音   435M
    # "dereverb_echo_mbr_fused.ckpt",  # 好，破音少了很多，但是有奇怪背景波动音          435M
    # "deverb_bs_roformer_8_384dim_10depth.ckpt",  # 好，破音少了很多，没有背景波动音           345M

    # "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",   # 好，破音少了很多，少少背景波动音    871M
    # "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",  # 好，破音少了很多，少少背景波动音        871M
    # "dereverb_mel_band_roformer_mono_anvuew.ckpt",  # 好，破音少了很多，少少背景波动音                                    871M

    "dereverb_mel_band_roformer_mono_anvuew.ckpt",
]
for model in models:
    separator = None
    try:
        separator = Separator(model_file_dir=os.path.join(util.get_home_dir(), '.cache', 'uvr'),
                              output_dir=os.path.join('output', model))
        separator.load_model(model_filename=model)
        output_files = separator.separate([
            '../../demo_jpn_single.wav',
            # 'Vocals.wav',
        ])
    except Exception as e:
        print(f'Failed to load model: {model}\nError: {e}')
    finally:
        print('Done')
