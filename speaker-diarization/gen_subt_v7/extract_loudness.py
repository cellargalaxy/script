import os
import util
import tool_loudness

logger = util.get_logger()


def extract_loudness(input_path, output_dir):
    output_path = os.path.join(output_dir, 'wav.wav')
    if util.path_exist(output_path):
        return output_path
    tool_loudness.loudness_normalization(input_path, output_path)
    return output_path


def exec(manager):
    logger.info("extract_loudness,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_loudness")
    output_path = extract_loudness(audio_path, output_dir)
    manager['extract_loudness_path'] = output_path
    manager['audio_path'] = output_path
    logger.info("extract_loudness,leave: %s", util.json_dumps(manager))
