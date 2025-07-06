import os

bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.5:10808'
os.environ['https_proxy'] = 'http://192.168.123.5:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.5,mirrors.ustc.edu.cn'

import util
import init
import extract_audio
import noise_reduction
import merge_channel
import part_detect
import part_divide
import part_split
import segment_detect
import segment_divide
import segment_split
import speaker_detect
import speaker_divide
import speaker_join

logger = util.get_logger()

manager = {
    "video_path": '../demo.mkv',
    "audio_track_index": 0,
    "auth_token": os.environ.get('auth_token', ''),
}

util.print_device_info()
init.exec(manager)

extract_audio.exec(manager)
noise_reduction.exec(manager)
merge_channel.exec(manager)

part_detect.exec(manager)
part_divide.exec(manager)
part_split.exec(manager)

segment_detect.exec(manager)
segment_divide.exec(manager)
segment_split.exec(manager)

speaker_detect.exec(manager)
speaker_divide.exec(manager)
