import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

import util
import pre_treatment
import noise_reduction_demucs
import audio_activity
import audio_batch
import audio_split
import gen_subt

logger = util.get_logger()

manager = {
    "video_path": "../long.mkv",
    "audio_track_index": 0,
    "auth_token": "",

    # "output_dir": "output/demo",
    # "audio_path": 'output/demo/extract_audio/wav.wav',
    # "noise_reduction_audio_path": 'output/demo/noise_reduction/htdemucs/wav/vocals.wav',
    # "merge_audio_channel_path": 'output/demo/merge_audio_channel/wav.wav',
    # "audio_activity_path": 'output/demo/audio_activity/audio_activity.json'
    # "audio_batch_path": 'output/demo/audio_batch/audio_batch.json'
    # "audio_split_dir": 'output/demo/audio_split'
    # "gen_subt_path": 'output/demo/gen_subt/gen_subt.json'
}

util.print_device_info()
pre_treatment.init_by_manager(manager)

pre_treatment.extract_audio_by_manager(manager)
noise_reduction_demucs.noise_reduction_by_manager(manager)
pre_treatment.merge_audio_channel_by_manager(manager)

audio_activity.audio_activity_by_manager(manager)
audio_batch.audio_batch_by_manager(manager)
audio_split.audio_split_by_manager(manager)
gen_subt.gen_subt_by_manager(manager)
