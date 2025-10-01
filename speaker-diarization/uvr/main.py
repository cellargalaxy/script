import os

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

logger = util.get_logger()


def exec(manager):
    util.print_device_info()
    init.exec(manager)

    extract_audio.exec(manager)
    extract_vocal.exec(manager)
    extract_main_vocal.exec(manager)
    extract_dereverb.exec(manager)


def exec_batch(video_paths):
    for i, video_path in enumerate(video_paths):
        manager = {
            "video_path": video_path,
            "audio_track_index": 0,
            "auth_token": os.environ.get('auth_token', ''),
        }
        try:
            exec(manager)
        except Exception:
            output_dir = manager.get('output_dir', None)
            if output_dir and len(output_dir) > 0 and util.path_exist(output_dir):
                util.delete_path(output_dir)
            exec(manager)


video_paths = [
    '../material/music/lanlianhua.flac',
]

exec_batch(video_paths)
