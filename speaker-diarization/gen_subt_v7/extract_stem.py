import util
import os
import extract_stem_uvr

logger = util.get_logger()


def exec(manager, names):
    logger.info("extract_stem,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_stem")
    path_map = extract_stem_uvr.extract_stem(audio_path, names, output_dir)
    manager['extract_stem_path_map'] = path_map
    manager['audio_path'] = path_map['audio_path']
    manager['split_audio_path'] = path_map['split_audio_path']
    logger.info("extract_stem,leave: %s", util.json_dumps(manager))
    util.exec_gc()
