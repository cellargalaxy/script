import tempfile
import numpy as np, soundfile as sf
import util
import os
from pydub import AudioSegment

logger = util.get_logger()

model = None


def get_model():
    from audiosr import build_model

    global model
    if model:
        return model
    model = build_model(model_name="basic", device=util.get_device_type())
    return model


def audiosr_audio(audio):
    from audiosr import super_resolution

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        input_path = tmp.name
        audio.export(input_path, format="wav")

    model = get_model()
    waveform = super_resolution(model, input_path, seed=42, guidance_scale=3.5, ddim_steps=50)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name
        out_wav = (waveform[0] * 32767).astype(np.int16).T
        sf.write(output_path, out_wav, samplerate=48000)
    audio = AudioSegment.from_wav(output_path)

    os.remove(input_path)
    os.remove(output_path)

    return audio


def split_audio(audio_path, json_path, output_dir, path_key, min_duration=None):
    wav_dir = os.path.join(output_dir, path_key)
    util.delete_path(wav_dir)
    audio = AudioSegment.from_wav(audio_path)
    segments = util.read_file_to_obj(json_path)
    for i, segment in enumerate(segments):
        if min_duration and segment['duration'] < min_duration:
            continue
        index = i
        if segment.get('index', None):
            index = segment.get('index', None)
        file_name = f"{index:04d}.wav"
        if segment.get('vad_type', ''):
            file_name = f"{index:04d}-{segment.get('vad_type', '')}.wav"
        if segment.get('text', ''):
            file_name = f"{index:04d}-{segment.get('text', '')}.wav"
        segments[i]['file_name'] = file_name
        cut_path = os.path.join(wav_dir, file_name)
        if segment.get('speaker', ''):
            cut_path = os.path.join(wav_dir, segment.get('speaker', ''), file_name)
        cut_path = util.truncate_path(cut_path)
        util.mkdir(cut_path)
        cut = audio[segment['start']:segment['end']]
        cut = audiosr_audio(cut)
        cut.export(cut_path, format='wav')
    json_path = os.path.join(output_dir, f'{path_key}.json')
    util.save_as_json(segments, json_path)


def exec(manager, path_key, min_duration=None):
    logger.info("split_audio_audiosr,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('split_audio_path')
    json_path = manager.get(path_key)
    output_dir = os.path.join(manager.get('output_dir'), 'split_audio_audiosr')
    split_audio(audio_path, json_path, output_dir, path_key, min_duration)
    logger.info("split_audio_audiosr,leave: %s", util.json_dumps(manager))


import sys

if __name__ == '__main__':
    args = {}
    for item in sys.argv[1:]:
        if '=' in item:
            k, v = item.split('=', 1)
            args[k] = v
    manager = util.json_loads(args['manager'])
    path_key = str(args['path_key'])
    min_duration = str(args.get('min_duration', ''))
    if min_duration.lower() in ['none', 'null', '']:
        min_duration = None
    exec(manager, path_key, min_duration)
