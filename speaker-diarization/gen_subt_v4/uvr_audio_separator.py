import util
import os
from audio_separator.separator import Separator

model_file_dir = os.path.join(util.get_home_dir(), '.cache', 'uvr')


def extract_vocal(audio_path, output_dir):
    output_dir = os.path.join(output_dir, 'vocal')
    vocal_path = os.path.join(output_dir, 'vocals.wav')
    if util.path_exist(vocal_path):
        return vocal_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename='vocals_mel_band_roformer.ckpt')
    output_names = {
        "Vocals": "vocals",
        "Other": "others",
    }
    separator.separate([audio_path], output_names)
    return vocal_path


def extract_main_vocal(audio_path, output_dir):
    output_dir = os.path.join(output_dir, 'main_vocal')
    main_vocal_path = os.path.join(output_dir, 'vocals.wav')
    if util.path_exist(main_vocal_path):
        return main_vocal_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename='mel_band_roformer_karaoke_becruily.ckpt')
    output_names = {
        "Vocals": "vocals",
        "Other": "others",
    }
    separator.separate([audio_path], output_names)
    return main_vocal_path


def uvr(audio_path, output_dir):
    vocal_path = extract_vocal(audio_path, output_dir)
    main_vocal_path = extract_vocal(vocal_path, output_dir)
    return main_vocal_path
