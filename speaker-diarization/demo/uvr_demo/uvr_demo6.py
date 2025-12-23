from audio_separator.separator import Separator
import util
import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

# 测试这所推荐的模型
# https://www.bilibili.com/video/BV1j42yBiERh/
models = [
    # "dereverb_mel_band_roformer_mono_anvuew.ckpt", # 混响
    "mel_band_roformer_instrumental_becruily.ckpt", # 和声
    "mel_band_roformer_vocals_becruily.ckpt",# 和声
    "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
]
for model in models:
    separator = None
    try:
        separator = Separator(model_file_dir=os.path.join(util.get_home_dir(), '.cache', 'uvr'),
                              output_dir=os.path.join('output', model))
        separator.load_model(model_filename=model)
        output_files = separator.separate([
            # '../../demo_jpn_single.wav',
            '/workspace/script/speaker-diarization/material/BOW_AND_ARROW.flac',
            # 'clover_wish.flac',
            # 'Credit_Theme.flac',
            # 'gang.mp3',
            # 'I_Really.flac',
            # 'innocent_arrogance.flac',
            # 'more_than_words.flac',
        ])
    except Exception as e:
        print(f'Failed to load model: {model}\nError: {e}')
    finally:
        print('Done')
