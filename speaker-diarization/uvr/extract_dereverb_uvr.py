import util
import os
from audio_separator.separator import Separator

model_file_dir = os.path.join(util.get_home_dir(), '.cache', 'uvr')


def extract_dereverb(audio_path, output_dir):
    output_path = os.path.join(output_dir, 'noreverb.wav')
    if util.path_exist(output_path):
        return output_path
    separator = Separator(model_file_dir=model_file_dir, output_dir=output_dir)
    separator.load_model(model_filename='dereverb_mel_band_roformer_mono_anvuew.ckpt')
    output_names = {
        "Noreverb": "noreverb",
        "Reverb": "reverb",
    }
    separator.separate([audio_path], output_names)
    return output_path
