import os
import util
import tool_loudness

logger = util.get_logger()


def extract_loudness(input_path_map, output_dir):
    output_path_map = {}
    for key, input_path in input_path_map.items():
        output_path = os.path.join(output_dir, util.get_file_basename(input_path))
        output_path_map[key] = output_path
        if util.path_exist(output_path):
            continue
        tool_loudness.loudness_normalization(input_path, output_path)
    return output_path_map


def exec(manager):
    logger.info("extract_loudness,enter: %s", util.json_dumps(manager))
    input_path_map = manager.get('extract_stem_path_map')
    output_dir = os.path.join(manager.get('output_dir'), "extract_loudness")
    output_path_map = extract_loudness(input_path_map, output_dir)
    manager['extract_loudness_path_map'] = output_path_map
    manager['audio_path'] = output_path_map['noreverb_path']
    logger.info("extract_loudness,leave: %s", util.json_dumps(manager))
    util.exec_gc()
