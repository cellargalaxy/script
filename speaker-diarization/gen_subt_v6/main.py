import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import util
import init
import extract_audio
import extract_vocal
import extract_main_vocal
import extract_dereverb
import merge_channel
import part_detect
import part_divide
import part_split
import segment_detect
import segment_divide
import segment_split
import speaker_segment
import speaker_join
import speaker_neigh
import speaker_overall

logger = util.get_logger()

video_paths = [
    '../holo/S01E01.mkv',
]

for i, video_path in enumerate(video_paths):
    manager = {
        "video_path": video_path,
        "audio_track_index": 0,
        "auth_token": os.environ.get('auth_token', ''),
    }

    util.print_device_info()
    init.exec(manager)

    extract_audio.exec(manager)
    extract_vocal.exec(manager)
    extract_main_vocal.exec(manager)
    extract_dereverb.exec(manager)
    merge_channel.exec(manager)

    part_detect.exec(manager)
    part_divide.exec(manager)
    part_split.exec(manager)

    segment_detect.exec(manager)
    segment_divide.exec(manager)
    segment_split.exec(manager)

    speaker_segment.exec(manager)
    speaker_join.exec(manager)
    speaker_neigh.exec(manager)
    speaker_overall.exec(manager)
