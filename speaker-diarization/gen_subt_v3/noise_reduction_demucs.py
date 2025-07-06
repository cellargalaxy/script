import demucs.separate
import os
import util

logger = util.get_logger()


def noise_reduction(audio_path, output_dir):
    output_path = os.path.join(output_dir, "htdemucs/wav/vocals.wav")
    if util.path_exist(output_path):
        return output_path
    cmd = [
        audio_path,
        '-v',
        '-d', util.get_device_type(),
        '-o', output_dir,
    ]
    demucs.separate.main(cmd)
    return output_path
