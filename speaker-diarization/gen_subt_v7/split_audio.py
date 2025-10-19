import util
import os
from pydub import AudioSegment

logger = util.get_logger()


def split_audio(audio_path, json_path, output_dir):
    util.mkdir(output_dir)
    audio = AudioSegment.from_wav(audio_path)
    segments = util.read_file_to_obj(json_path)
    for i, segment in enumerate(segments):
        file_name = f"{i:03d}.wav"
        if segment.get('vad_type', ''):
            file_name = f"{i:03d}-{segment.get('vad_type', '')}.wav"
        if segment.get('speaker', ''):
            file_name = f"{i:03d}-{segment.get('speaker', '')}.wav"
        if segment.get('text', ''):
            file_name = f"{i:03d}-{segment.get('text', '')}.wav"
        cut_path = os.path.join(output_dir, file_name)
        cut = audio[segment['start']:segment['end']]
        cut.export(cut_path, format='wav')


def exec(manager, path_key):
    logger.info("split_audio,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    json_path = manager.get(path_key)
    output_dir = os.path.join(manager.get('output_dir'), 'split_audio', path_key)
    split_audio(audio_path, json_path, output_dir)
    logger.info("split_audio,leave: %s", util.json_dumps(manager))
