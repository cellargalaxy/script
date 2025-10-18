import util
import os
import separate_stem_uvr

logger = util.get_logger()


def exec(manager):
    logger.info("separate_stem,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "separate_stem")
    audio_path = separate_stem_uvr.separate_stem(audio_path, output_dir)
    manager['audio_path'] = audio_path
    logger.info("separate_stem,leave: %s", util.json_dumps(manager))
