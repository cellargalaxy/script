import util
import json
import os
from pydub import AudioSegment

logger = util.get_logger()


def segment_split(audio_path, segment_divide_path, output_dir):
    if util.path_exist(output_dir):
        return

    content = util.read_file(segment_divide_path)
    segments = json.loads(content)
    util.mkdir(output_dir)
    audio = AudioSegment.from_wav(audio_path)
    for i, segment in enumerate(segments):
        cut_path = os.path.join(output_dir, f'{i:05d}_{segment["vad_type"]}.wav')
        cut = audio[segment['start']:segment['end']]
        cut.export(cut_path, format="wav")


def segment_split_by_manager(manager):
    logger.info("segment_split,enter,manager: %s", json.dumps(manager))
    audio_path = manager.get('merge_audio_channel_path')
    segment_divide_path = manager.get('segment_divide_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_split")
    segment_split(audio_path, segment_divide_path, output_dir)
    manager['segment_split_dir'] = output_dir
    logger.info("segment_split,leave,manager: %s", json.dumps(manager))
