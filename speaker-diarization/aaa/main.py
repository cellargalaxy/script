import util
import pre_treatment
import noise_reduction_demucs
import audio_activity

import os

os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

os.environ['TORCH_HOME'] = os.path.join(util.get_script_path(), "model/torch")

logger = util.get_logger()

manager = {
    "video_path": "../demo.mkv",
    "audio_track_index": 0,
    "auth_token": "",

    "output_dir": "output/demo",
    "audio_path": 'output/demo/extract_audio/wav.wav',
    "noise_reduction_audio_path": 'output/demo/noise_reduction/htdemucs/wav/vocals.wav',
}

util.print_device_info()
pre_treatment.init_by_manager(manager)

pre_treatment.extract_audio_by_manager(manager)
noise_reduction_demucs.noise_reduction_by_manager(manager)

audio_activity.audio_activity_by_manager(manager)
