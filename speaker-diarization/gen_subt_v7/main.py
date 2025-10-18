import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import util
import init
import separate_audio

logger = util.get_logger()


def exec(manager):
    init.exec(manager)
    extract_audio.exec(manager)


def exec_batch(video_paths):
    for i, video_path in enumerate(video_paths):
        try:
            manager = {
                "video_path": video_path,
            }
            exec(manager)
        except Exception as e:
            logger.error("exec_batch failed.", exc_info=True)
            util.input_timeout("异常，回车继续: ", 60)


video_paths = [
    '../material/mao.mp3',
]

exec_batch(video_paths)
