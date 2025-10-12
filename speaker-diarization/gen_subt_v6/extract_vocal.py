import os
import util
import extract_vocal_uvr

logger = util.get_logger()


def exec(manager):
    logger.info("extract_vocal,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_vocal")
    output_path = extract_vocal_uvr.extract_vocal(audio_path, output_dir)
    manager['extract_vocal_path'] = output_path
    manager['audio_path'] = audio_path
    logger.info("extract_vocal,leave: %s", util.json_dumps(manager))
    util.exec_gc()
