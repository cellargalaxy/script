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
# https://r1kc63iz15l.feishu.cn/wiki/Dy0bwG4XIizBgJkePDucILaMnlf
models = [
    # 声轨提取，已测过
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",  # 分离人声伴奏推荐使用！
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # 1297提取出来的音频极高频有点问题，但是SDR值高，可以使用
    # 声轨提取，未测过，在库模型
    "big_beta5e.ckpt",  # 超级大模型，用于提取人声，处理速度和出来的质量都很不错，但出来的人声存在少量噪声
    "mel_band_roformer_instrumental_becruily.ckpt",  # 针对提取伴奏训练的模型，提取出来的伴奏SDR值和1296提取的一样，但是推理速度更快
    "mel_band_roformer_vocals_becruily.ckpt",  # 针对提取人声训练的模型，处理速度和出来的质量都很不错，但出来的人声存在少量噪声
    "melband_roformer_inst_v2.ckpt",  # 超级大模型，如果上面的inst_v1e你不满意的话，来试试这个v2的伴奏提取模型吧
    "melband_roformer_instvox_duality_v2.ckpt",  # 平衡了人声和伴奏提取的质量，并且处理速度较快，虽然他的模型大小最大
    # 声轨提取，未测过，离库模型
    "HTDemucs4_6stems.th",  # 提取多轨可以使用，拆的最多，一共6轨，最推荐
    "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",  # 分离和声模型，如果是原曲放进去，出来的就是带和声伴奏
    "Kim_MelBandRoformer.ckpt",  # Kim的模型，虽然效果比1296差一点，但是速度更快
    "BS-Roformer_LargeV1.ckpt",  # 也是一个大模型，用于提取人声，处理速度和出来的质量都很不错，能比得上bsr1296
    "inst_v1e.ckpt",  # 针对提取伴奏训练的模型，出来的伴奏非常接近原版伴奏
    "kimmel_unwa_ft.ckpt",  # Kim_MelBandRoformer的微调模型，处理速度较快，并且分离表现很不错

    # 去混响，基本都测过了，我的最好
    # "deverb_bs_roformer_8_256dim_8depth.ckpt",  #去混响首选！也能去除部分和声！但是无法去延迟
    # "deverb_bs_roformer_8_384dim_10depth.ckpt",  #比上面那个激进一点点，不过基本听不出差别
    # "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",  #分离混响推荐！不支持分离单声道混响
    # "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",  #比上面那个少激进一点，不支持分离单声道混响

    # 降噪，都测过
    # "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",  # 最新的降噪模型，推荐！
    # "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",  # 比上面的降噪激进一点
]
for model in models:
    separator = None
    try:
        separator = Separator(model_file_dir=os.path.join(util.get_home_dir(), '.cache', 'uvr'),
                              output_dir=os.path.join('output', model))
        separator.load_model(model_filename=model)
        output_files = separator.separate([
            '../../demo_jpn_single.wav',
            # 'BOW_AND_ARROW.flac',
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
