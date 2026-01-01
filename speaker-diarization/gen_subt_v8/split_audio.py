import util
import os
from pydub import AudioSegment
import tool_loudness

logger = util.get_logger()


def split_audio(audio_path, json_path, output_dir, path_key, min_duration=None):
    wav_dir = os.path.join(output_dir, path_key)
    util.delete_path(wav_dir)
    audio = AudioSegment.from_wav(audio_path)
    segments = util.read_file_to_obj(json_path)
    for i, segment in enumerate(segments):
        if segment['vad_type'] == 'silence':
            continue
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
        cut = tool_loudness.get_loudness(cut)
        cut.export(cut_path, format='wav')
    json_path = os.path.join(output_dir, f'{path_key}.json')
    util.save_as_json(segments, json_path)


def exec(manager, path_key, min_duration=None):
    logger.info("split_audio,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('split_audio_path')
    json_path = manager.get(path_key)
    output_dir = os.path.join(manager.get('output_dir'), 'split_audio')
    split_audio(audio_path, json_path, output_dir, path_key, min_duration)
    logger.info("split_audio,leave: %s", util.json_dumps(manager))
