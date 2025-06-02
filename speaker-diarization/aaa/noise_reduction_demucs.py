import demucs.separate
import os
import util


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
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "noise_reduction")
    output_path = noise_reduction(audio_path, output_dir)
    manager['noise_reduction_audio_path'] = output_path
