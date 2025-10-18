import util
import util_ffmpeg
import os

logger = util.get_logger()


def extract_simple(input_path, output_dir):
    output_path = os.path.join(output_dir, "wav.wav")
    if util.path_exist(output_path):
        return output_path
    util_ffmpeg.simple_audio(input_path, output_path)
    return output_path


def exec(manager):
    logger.info("extract_simple,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('extract_stem_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_simple")
    audio_path = extract_simple(audio_path, output_dir)
    manager['extract_simple_path'] = audio_path
    logger.info("extract_simple,leave: %s", util.json_dumps(manager))
