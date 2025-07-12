import os
import util
import json
import uvr_audio_separator

logger = util.get_logger()


def exec(manager):
    logger.info("uvr,enter: %s", json.dumps(manager))
    audio_path = manager.get('extract_audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "uvr")
    output_path = uvr_audio_separator.uvr(audio_path, output_dir)
    manager['uvr_path'] = output_path
    logger.info("uvr,leave: %s", json.dumps(manager))
    util.exec_gc()
