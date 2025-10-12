import os
import util
import extract_dereverb_uvr

logger = util.get_logger()


def exec(manager):
    logger.info("extract_dereverb,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_dereverb")
    output_path = extract_dereverb_uvr.extract_dereverb(audio_path, output_dir)
    manager['extract_dereverb_path'] = output_path
    manager['audio_path'] = audio_path
    logger.info("extract_dereverb,leave: %s", util.json_dumps(manager))
    util.exec_gc()
