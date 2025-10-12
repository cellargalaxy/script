import os
import util
import extract_main_vocal_uvr

logger = util.get_logger()


def exec(manager):
    logger.info("extract_main_vocal,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_main_vocal")
    output_path = extract_main_vocal_uvr.extract_main_vocal(audio_path, output_dir)
    manager['extract_main_vocal_path'] = output_path
    manager['audio_path'] = output_path
    logger.info("extract_main_vocal,leave: %s", util.json_dumps(manager))
    util.exec_gc()
