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
import activity_detect
import part_divide
import part_split
import part_activity_detect
import speaker_detect
import subt_gen

logger = util.get_logger()

manager = {
    "video_path": '../long.mkv',
    "audio_track_index": 0,
    "auth_token": os.environ.get('auth_token', ''),
}

util.print_device_info()
pre_treatment.init_by_manager(manager)

pre_treatment.extract_audio_by_manager(manager)
noise_reduction_demucs.noise_reduction_by_manager(manager)
pre_treatment.merge_audio_channel_by_manager(manager)

activity_detect.activity_detect_by_manager(manager)
part_divide.part_divide_by_manager(manager)
part_split.part_split_by_manager(manager)

# part_activity_detect.part_activity_detect_by_manager(manager)
# speaker_detect.speaker_detect_by_manager(manager)
subt_gen.subt_gen_by_manager(manager)
