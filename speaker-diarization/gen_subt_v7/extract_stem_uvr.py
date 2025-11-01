import util
import os
from audio_separator.separator import Separator
from pydub import AudioSegment
import part_detect_vad
import math
import tool_subt

model_file_dir = os.path.join(util.get_home_dir(), '.cache', 'uvr')


class Extract:
    def __init__(self, handler):
        self.handler = handler

    def extract(self, audio_path, output_dir):
        master_path, slave_path = extract_part(self.handler, audio_path, output_dir)
        self.master_path = master_path
        self.slave_path = slave_path
        self.master_copy_path = os.path.join(output_dir, self.handler.get_master_name())
        self.slave_copy_path = os.path.join(output_dir, self.handler.get_slave_name())

    def copy_master(self):
        util.copy_file(self.master_path, self.master_copy_path)

    def copy_slave(self):
        util.copy_file(self.slave_path, self.slave_copy_path)


def part_divide(audio_path, output_dir, segments=None):
    if not segments:
        audio = AudioSegment.from_wav(audio_path)
        segments = part_detect_vad.part_detect_by_data(audio)
    util.save_as_json(segments, os.path.join(output_dir, 'original.json'))
    tool_subt.save_segments_as_srt(segments, os.path.join(output_dir, 'original.srt'))

    results = [{'start': 0, 'end': 0}]
    for i, segment in enumerate(segments):
        results[-1]['end'] = segment['end']
        if results[-1]['end'] - results[-1]['start'] < 1000 * 60 * 5:
            continue
        if segment['vad_type'] != 'silence':
            continue
        if segment['duration'] < 1000:
            continue
        mean = math.floor((segment['start'] + segment['end']) / 2.0)
        results[-1]['end'] = mean
        results.append({'start': mean, 'end': segment['end']})
    if len(results) > 1 and results[-1]['end'] - results[-1]['start'] < 1000 * 60 * 5:
        results[-2]['end'] = results[-1]['end']
        results.pop()
    results = tool_subt.fix_overlap_segments(results)
    results = tool_subt.init_segments(results)
    tool_subt.check_coherent_segments(results)
    util.save_as_json(results, os.path.join(output_dir, 'part.json'))
    tool_subt.save_segments_as_srt(results, os.path.join(output_dir, 'part.srt'))
    return results


def extract_part(handler, audio_path, output_dir, segments=None):
    name = handler.get_name()
    master_path = os.path.join(output_dir, name, "master.wav")
    slave_path = os.path.join(output_dir, name, "slave.wav")
    if util.path_exist(master_path) and util.path_exist(slave_path):
        return master_path, slave_path

    segments = part_divide(audio_path, output_dir, segments)

    audio = AudioSegment.from_wav(audio_path)
    master_paths = []
    slave_paths = []
    for i, segment in enumerate(segments):
        index = i
        if segment.get('index', None):
            index = segment.get('index', None)
        cut_path = os.path.join(output_dir, name, 'part', f"{index:04d}.wav")
        util.mkdir(cut_path)
        cut = audio[segment['start']:segment['end']]
        cut.export(cut_path, format='wav')

        cut_dir = os.path.join(output_dir, name, 'part', f"{index:04d}")
        master, slave = handler.extract(cut_path, cut_dir)
        master_paths.append(master)
        slave_paths.append(slave)

    master_audio = AudioSegment.empty()
    for i, path in enumerate(master_paths):
        segment = AudioSegment.from_wav(path)
        master_audio += segment
    master_audio.export(master_path, format='wav')

    slave_audio = AudioSegment.empty()
    for i, path in enumerate(slave_paths):
        segment = AudioSegment.from_wav(path)
        slave_audio += segment
    slave_audio.export(slave_path, format='wav')

    return master_path, slave_path


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


def new_extract(names):
    extracts = []
    for i, name in enumerate(names):
        if name == "vocal":
            extracts.append(Extract(VocalHandler()))
        if name == "main_vocal":
            extracts.append(Extract(MainVocalHandler()))
        if name == "dereverb":
            extracts.append(Extract(DeReverbHandler()))
    return extracts


def extract_stem(audio_path, names, output_dir):
    path_map = {}
    extracts = new_extract(names)
    for i, extract in enumerate(extracts):
        extract.extract(audio_path, output_dir)
        extract.copy_slave()
        slave_copy_path = extract.slave_copy_path
        file_name = util.get_file_name(slave_copy_path)
        path_map[f'{file_name}_path'] = slave_copy_path
        audio_path = extract.master_path
    if len(extracts) > 0:
        extracts[-1].copy_master()
        master_copy_path = extracts[-1].master_copy_path
        file_name = util.get_file_name(master_copy_path)
        path_map[f'{file_name}_path'] = master_copy_path
        path_map['audio_path'] = master_copy_path
        path_map['split_audio_path'] = master_copy_path
    return path_map
