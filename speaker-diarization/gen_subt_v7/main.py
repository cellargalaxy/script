import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

import util

logger = util.get_logger()


def exec(manager):
    import init
    init.exec(manager)
    import extract_audio
    extract_audio.exec(manager)
    import extract_stem
    extract_stem.exec(manager)
    import extract_simple
    extract_simple.exec(manager)
    import extract_loudness
    extract_loudness.exec(manager)
    import part_detect
    part_detect.exec(manager)

    import split_audio
    split_audio.exec(manager, 'part_detect_path')


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
