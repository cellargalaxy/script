import util
import os
from audio_separator.separator import Separator

model_file_dir = os.path.join(util.get_home_dir(), '.cache', 'uvr')


class VocalHandler:
    def get_name(self):
        return "vocal"

    def get_master_name(self):
        return "vocal.wav"

    def get_slave_name(self):
        return "bgm.wav"

    def extract(self, audio_path, output_dir):
        vocals_path, others_path = extract_vocal(audio_path, output_dir)
        return vocals_path, others_path


def extract_vocal(audio_path, output_dir):
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


class MainVocalHandler:
    def get_name(self):
        return "main_vocal"

    def get_master_name(self):
        return "main_vocal.wav"

    def get_slave_name(self):
        return "harmony.wav"

    def extract(self, audio_path, output_dir):
        vocals_path, instrumental_path = extract_main_vocal(audio_path, output_dir)
        return vocals_path, instrumental_path


def extract_main_vocal(audio_path, output_dir):
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


class DeReverbHandler:
    def get_name(self):
        return "dereverb"

    def get_master_name(self):
        return "noreverb.wav"

    def get_slave_name(self):
        return "reverb.wav"

    def extract(self, audio_path, output_dir):
        noreverb_path, reverb_path = extract_dereverb(audio_path, output_dir)
        return noreverb_path, reverb_path


def extract_dereverb(audio_path, output_dir):
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
