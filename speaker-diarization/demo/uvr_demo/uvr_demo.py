from audio_separator.separator import Separator
import util
import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

# 主唱提取
models = [
    # "UVR_MDXNET_Main.onnx",  # 和声完全没有被去掉
    # "melband_roformer_big_beta4.ckpt",  # 和声完全没有被去掉
    # "vocals_mel_band_roformer.ckpt",  # 和声完全没有被去掉
    # "model_bs_roformer_ep_368_sdr_12.9628.ckpt",  # 和声完全没有被去掉
    # "MDX23C-8KFFT-InstVoc_HQ_2.ckpt",  # 和声完全没有被去掉
    # "MDX23C-8KFFT-InstVoc_HQ.ckpt",  # 和声完全没有被去掉
    # "UVR-MDX-NET-Voc_FT.onnx",  # 和声完全没有被去掉
    # "Kim_Vocal_2.onnx",  # 和声完全没有被去掉
    # "kuielab_b_vocals.onnx",  # 和声完全没有被去掉
    # "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # 什么声音都没有
    # "UVR-MDX-NET_Main_438.onnx",  # 和声完全没有被去掉
    # "kuielab_a_vocals.onnx",  # 和声完全没有被去掉
    # "MelBandRoformerBigSYHFTV3Epsilon.ckpt",
    # "MelBandRoformerBigSYHFTV2.5.ckpt",
    # "MelBandRoformerBigSYHFTV2.ckpt",
    # "MelBandRoformerBigSYHFTV1.ckpt",  # 和声完全没有被去掉
    # "MelBandRoformerSYHFTV3Epsilon.ckpt",  # 去除了过多人声，且有人声也没有去掉和声
    # "MelBandRoformerSYHFT.ckpt",  # 去除了过多人声，且有人声也没有去掉和声
    # "melband_roformer_instvoc_duality_v2.ckpt",
    # "melband_roformer_instvoc_duality_v1.ckpt",  # 和声完全没有被去掉
    # "4_HP-Vocal-UVR.pth",  # 和声完全没有被去掉
    # "3_HP-Vocal-UVR.pth",  # 和声完全没有被去掉
    # "1_HP-UVR.pth",  # 和声完全没有被去掉
    # "mel_band_roformer_kim_ft_unwa.ckpt", # 和声完全没有被去掉
    # "8_HP2-UVR.pth",  # 和声完全没有被去掉
    # "7_HP2-UVR.pth",  # 和声完全没有被去掉
    # "2_HP-UVR.pth", # 和声完全没有被去掉
    # "17_HP-Wind_Inst-UVR.pth",# 和声完全没有被去掉

    # "HP2_all_vocals.pth",
    # "HP3_all_vocals.pth",
    # "HP5_only_main_vocal.pth",
    # "HP5++.pth",
    # "HP2++.pth",

    # "UVR-DeEcho-DeReverb.pth",
    # "UVR-De-Reverb-aufr33-jarredou.pth",
    # "ereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
    # "MDX23C-De-Reverb-aufr33-jarredou.ckpt",
    # "Reverb_HQ_By_FoxJoy.onnx",
    # "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"
    #
    # "5_HP-Karaoke-UVR.pth", # 安装依赖：sudo apt-get install -y libsamplerate0 libsamplerate0-dev
    # "6_HP-Karaoke-UVR.pth",
    # "9_HP2-UVR.pth",  # 减淡了和声
    # "16_SP-UVR-MID-44100-2.pth",  # 减淡了和声
    # "UVR_MDXNET_KARA.onnx", # 有漏，压的没有5_HP大
    # "UVR_MDXNET_KARA_2.onnx", # 比5_HP好，但是压的也更大

    # "UVR-MDX-NET-Inst_HQ_3.onnx",  # 和声完全没有被去掉
    # "UVR-MDX-NET-Inst_HQ_4.onnx", # 有漏
    # "UVR-MDX-NET-Inst_HQ_5.onnx", # 和声完全没有被去掉
    # "UVR-MDX-NET-Inst_Main.onnx", # 和声完全没有被去掉

    # "melband_roformer_big_beta5e.ckpt",  # 和声完全没有被去掉
    # "mel_band_roformer_instrumental_becruily.ckpt",  # 和声完全没有被去掉
    # "mel_band_roformer_vocals_becruily.ckpt",  # 和声完全没有被去掉
    # "melband_roformer_inst_v2.ckpt",  # 和声完全没有被去掉
    # "melband_roformer_instvox_duality_v2.ckpt",  # 和声完全没有被去掉

    # "htdemucs_ft.yaml",  # 和声完全没有被去掉
    # "htdemucs.yaml",  # 和声完全没有被去掉
    # "hdemucs_mmi.yaml",  # 和声完全没有被去掉
    # "htdemucs_6s.yaml",  # 和声完全没有被去掉
    # "model_bs_roformer_ep_317_sdr_12.9755.ckpt",  # 和声完全没有被去掉
    # "model_bs_roformer_ep_937_sdr_10.5309.ckpt",  # 和声完全没有被去掉
    # "melband_roformer_big_beta6.ckpt",  # 和声完全没有被去掉
    # "melband_roformer_big_beta6x.ckpt",  # 和声完全没有被去掉
    # "melband_roformer_inst_v1_plus.ckpt",  # 和声完全没有被去掉
    # "melband_roformer_inst_v1e.ckpt",  # 和声完全没有被去掉
    # "melband_roformer_inst_v1e_plus.ckpt",  # 和声完全没有被去掉
    # "melband_roformer_inst_v1.ckpt",  # 和声完全没有被去掉

    # "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",  # 好
    # "mel_band_roformer_karaoke_gabox.ckpt",  # 好
    # "mel_band_roformer_karaoke_becruily.ckpt",  # 好

    "mel_band_roformer_karaoke_becruily.ckpt",
]
for model in models:
    separator = None
    try:
        print(model)
        separator = Separator(model_file_dir=os.path.join(util.get_home_dir(), '.cache', 'uvr'),
                              output_dir=os.path.join('output', model))
        separator.load_model(model_filename=model)
        output_files = separator.separate([
            '/workspace/script/speaker-diarization/material/mao.mp3',
            # '/workspace/script/speaker-diarization/material/demo.mkv',
            # '../../material/demo.wav',
            # '../../material/BOW_AND_ARROW.flac',
            # '../../material/clover_wish.flac',
            # '../../material/Credit_Theme.flac',
            # '../../material/I_Really.flac',
            # '../../material/more_than_words.flac',
        ])
    except Exception as e:
        print(f'Failed to load model: {model}\nError: {e}')
        # finally:
        print('Done')
