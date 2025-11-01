import util
import tool_ffmpeg
import os

logger = util.get_logger()


def extract_simple(input_path, output_dir):
    output_path = os.path.join(output_dir, "wav.wav")
    if util.path_exist(output_path):
        return output_path
    tool_ffmpeg.extract_simple_audio(input_path, output_path)
    return output_path


def exec(manager):
    logger.info("extract_simple,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_simple")
    output_path = extract_simple(audio_path, output_dir)
    manager['extract_simple_path'] = output_path
    manager['audio_path'] = output_path
    manager['split_audio_path'] = output_path
    logger.info("extract_simple,leave: %s", util.json_dumps(manager))
    util.exec_gc()
