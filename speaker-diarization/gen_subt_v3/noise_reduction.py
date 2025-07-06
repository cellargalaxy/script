import demucs.separate
import os
import util
import json
import noise_reduction_demucs

logger = util.get_logger()


def exec(manager):
    logger.info("noise_reduction,enter: %s", json.dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "noise_reduction")
    output_path = noise_reduction_demucs.noise_reduction(audio_path, output_dir)
    manager['noise_reduction_path'] = output_path
    logger.info("noise_reduction,leave: %s", json.dumps(manager))
