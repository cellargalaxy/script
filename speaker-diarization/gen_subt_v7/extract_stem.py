import util
import os
import extract_stem_uvr

logger = util.get_logger()


def exec(manager):
    logger.info("extract_stem,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('extract_audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_stem")
    audio_path = extract_stem_uvr.extract_stem(audio_path, output_dir)
    manager['extract_stem_path'] = audio_path
    logger.info("extract_stem,leave: %s", util.json_dumps(manager))
