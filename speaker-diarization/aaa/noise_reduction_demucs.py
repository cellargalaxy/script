import demucs.separate
import os
import util
import json

logger = util.get_logger()


def noise_reduction(audio_path, output_dir):
    cmd = [
        audio_path,
        '-v',
        '-d', util.get_device_type(),
        '-o', output_dir,
    ]
    demucs.separate.main(cmd)
    util.exec_gc()
    output_path = os.path.join(output_dir, "htdemucs/wav/vocals.wav")
    return output_path


def noise_reduction_by_manager(manager):
    logger.info("noise_reduction,enter,manager: %s", json.dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "noise_reduction")
    util.delete_path(output_dir)
    output_path = noise_reduction(audio_path, output_dir)
    manager['noise_reduction_audio_path'] = output_path
    logger.info("noise_reduction,leave,manager: %s", json.dumps(manager))
