import util
import json
import os
from pydub import AudioSegment

logger = util.get_logger()


def audio_split(audio_path, audio_batch_path, output_dir):
    if util.path_exist(output_dir):
        return

    content = util.read_file(audio_batch_path)
    segments = json.loads(content)
    util.mkdir(output_dir)
    audio = AudioSegment.from_wav(audio_path)
    for i, segment in enumerate(segments):
        cut_path = os.path.join(output_dir, f'{i:05d}_{segment.get("vad_type", "speech")}.wav')
        cut = audio[segment['start']:segment['end']]
        cut.export(cut_path, format="wav")


def audio_split_by_manager(manager):
    logger.info("audio_split,enter,manager: %s", json.dumps(manager))
    audio_path = manager.get('noise_reduction_audio_path')
    audio_batch_path = manager.get('audio_batch_path')
    output_dir = os.path.join(manager.get('output_dir'), "audio_split")
    util.delete_path(output_dir)
    audio_split(audio_path, audio_batch_path, output_dir)
    manager['audio_split_dir'] = output_dir
    logger.info("audio_split,leave,manager: %s", json.dumps(manager))
