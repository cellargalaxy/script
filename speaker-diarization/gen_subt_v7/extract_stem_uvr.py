import util
import os
from audio_separator.separator import Separator

model_file_dir = os.path.join(util.get_home_dir(), '.cache', 'uvr')


def extract_vocal(audio_path, output_dir):
    output_dir = os.path.join(output_dir, 'vocal')
    vocals_path = os.path.join(output_dir, 'vocals.wav')
    others_path = os.path.join(output_dir, 'others.wav')
    if util.path_exist(vocals_path) and util.path_exist(others_path):
        return vocals_path, others_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename='vocals_mel_band_roformer.ckpt')
    output_names = {
        "Vocals": "vocals",
        "Other": "others",
    }
    separator.separate([audio_path], output_names)
    return vocals_path, others_path


def extract_main_vocal(audio_path, output_dir):
    output_dir = os.path.join(output_dir, 'main_vocal')
    vocals_path = os.path.join(output_dir, 'vocals.wav')
    instrumental_path = os.path.join(output_dir, 'instrumental.wav')
    if util.path_exist(vocals_path) and util.path_exist(instrumental_path):
        return vocals_path, instrumental_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename='mel_band_roformer_karaoke_becruily.ckpt')
    output_names = {
        "Vocals": "vocals",
        "Instrumental": "instrumental",
    }
    separator.separate([audio_path], output_names)
    return vocals_path, instrumental_path


def extract_dereverb(audio_path, output_dir):
    output_dir = os.path.join(output_dir, 'dereverb')
    noreverb_path = os.path.join(output_dir, 'noreverb.wav')
    reverb_path = os.path.join(output_dir, 'reverb.wav')
    if util.path_exist(noreverb_path) and util.path_exist(reverb_path):
        return noreverb_path, reverb_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename='dereverb_mel_band_roformer_mono_anvuew.ckpt')
    output_names = {
        "Noreverb": "noreverb",
        "Reverb": "reverb",
    }
    separator.separate([audio_path], output_names)
    return noreverb_path, reverb_path


def extract_stem(audio_path, output_dir):
    output_path = os.path.join(output_dir, 'noreverb.wav')
    if util.path_exist(output_path):
        return output_path

    vocal_path, bgm_path = extract_vocal(audio_path, output_dir)
    util.copy_file(bgm_path, os.path.join(output_dir, 'bgm.wav'))

    main_vocal_path, harmony_path = extract_main_vocal(vocal_path, output_dir)
    util.copy_file(harmony_path, os.path.join(output_dir, 'harmony.wav'))

    noreverb_path, reverb_path = extract_dereverb(main_vocal_path, output_dir)
    util.copy_file(noreverb_path, os.path.join(output_dir, 'noreverb.wav'))
    util.copy_file(reverb_path, os.path.join(output_dir, 'reverb.wav'))

    return output_path
