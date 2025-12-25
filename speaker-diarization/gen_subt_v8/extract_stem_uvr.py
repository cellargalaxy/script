import util
import os
from audio_separator.separator import Separator
import extract_stem

model_file_dir = os.path.join(util.get_home_dir(), '.cache', 'uvr')

logger = util.get_logger()


class VocalHandler:
    def __init__(self, model_name=None):
        if not model_name:
            model_name = 'vocals_mel_band_roformer.ckpt'
        self.model_name = model_name

    def get_name(self):
        return "vocal"

    def get_master_name(self):
        return "vocal.wav"

    def get_slave_name(self):
        return "bgm.wav"

    def extract(self, audio_path, output_dir):
        output_dir = os.path.join(output_dir, self.model_name)
        vocals_path, others_path = extract_vocal(self.model_name, audio_path, output_dir)
        return vocals_path, others_path


def extract_vocal(model_name, audio_path, output_dir):
    vocals_path = os.path.join(output_dir, 'vocals.wav')
    others_path = os.path.join(output_dir, 'others.wav')
    done_path = extract_stem.done_path_ctx.get()
    if done_path and util.path_exist(done_path):
        return vocals_path, others_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename=model_name)
    output_names = {
        "Vocals": "vocals",
        "Other": "others",
    }
    separator.separate([audio_path], output_names)
    return vocals_path, others_path


class MainVocalHandler:
    def __init__(self, model_name=None):
        if not model_name:
            model_name = 'bs_roformer_karaoke_frazer_becruily.ckpt'
        self.model_name = model_name

    def get_name(self):
        return "main_vocal"

    def get_master_name(self):
        return "main_vocal.wav"

    def get_slave_name(self):
        return "harmony.wav"

    def extract(self, audio_path, output_dir):
        output_dir = os.path.join(output_dir, self.model_name)
        vocals_path, instrumental_path = extract_main_vocal(self.model_name, audio_path, output_dir)
        return vocals_path, instrumental_path


def extract_main_vocal(model_name, audio_path, output_dir):
    vocals_path = os.path.join(output_dir, 'vocals.wav')
    instrumental_path = os.path.join(output_dir, 'instrumental.wav')
    done_path = extract_stem.done_path_ctx.get()
    if done_path and util.path_exist(done_path):
        return vocals_path, instrumental_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename=model_name)
    output_names = {
        "Vocals": "vocals",
        "Instrumental": "instrumental",
    }
    separator.separate([audio_path], output_names)
    return vocals_path, instrumental_path


class DeReverbHandler:
    def __init__(self, model_name=None):
        if not model_name:
            # MDX23C-De-Reverb-aufr33-jarredou.ckpt
            model_name = 'dereverb_mel_band_roformer_mono_anvuew.ckpt'
        self.model_name = model_name

    def get_name(self):
        return "dereverb"

    def get_master_name(self):
        return "noreverb.wav"

    def get_slave_name(self):
        return "reverb.wav"

    def extract(self, audio_path, output_dir):
        output_dir = os.path.join(output_dir, self.model_name)
        noreverb_path, reverb_path = extract_dereverb(self.model_name, audio_path, output_dir)
        return noreverb_path, reverb_path


def extract_dereverb(model_name, audio_path, output_dir):
    noreverb_path = os.path.join(output_dir, 'noreverb.wav')
    reverb_path = os.path.join(output_dir, 'reverb.wav')
    done_path = extract_stem.done_path_ctx.get()
    if done_path and util.path_exist(done_path):
        return noreverb_path, reverb_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename=model_name)
    output_names = {
        "Noreverb": "noreverb",
        "Reverb": "reverb",
    }
    if model_name in ['MDX23C-De-Reverb-aufr33-jarredou.ckpt']:
        output_names = {
            "dry": "noreverb",
            "No dry": "reverb",
        }
    separator.separate([audio_path], output_names)
    return noreverb_path, reverb_path
