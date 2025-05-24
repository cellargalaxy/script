import json
import time

import util

logger = util.get_logger()


def get_audio_track_info(video_path):
    """
    :param video_path:
    :return: [{"index": 1, "tags": {"language": "jpn"}}, {"index": 2, "tags": {"language": "eng"}}]
    """
    cmd = [
        'bin/ffprobe',
        '-v', 'error',  # -v 是 --loglevel 的缩写，error 表示只输出错误信息，屏蔽警告和冗余信息，保持输出干净
        '-select_streams', 'a',  # 只处理“音频流（audio streams）”。a 表示所有音频轨道。可用值还有 v（视频），s（字幕）等
        '-show_entries', 'stream',  # 只输出 stream 中的 index,language 字段
        '-of', 'json',  # of 是 output format，这里指定为 json，便于在 Python 中解析处理。常见格式还有 default、csv 等
        video_path
    ]
    logger.info("查询视频文件音轨,cmd: %s", json.dumps(cmd))
    stdout, return_code = util.run_cmd(cmd)
    if return_code != 0:
        logger.error("查询视频文件音轨，异常")
        raise ValueError("查询视频文件音轨，异常")
    info = json.loads(stdout)
    logger.info("查询视频文件音轨,info: %s", json.dumps(info))
    audio_streams = info.get('streams', [])
    return audio_streams


def choice_audio_track_info_in_terminal_by_manager(manager):
    audio_streams = get_audio_track_info(manager.get('video_path', ''))
    logger.info("音轨列表: %s", json.dumps(audio_streams))
    for stream in audio_streams:
        logger.info("音轨列表,%s:%s", stream.get('index', 0) - 1, stream.get('tags', {}).get('language', '未知'))
    time.sleep(0.5)
    index = int(input("输入音轨编号: "))
    manager['audio_track_index'] = index


def get_video_duration(video_path):
    cmd = [
        'bin/ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    logger.info("查询视频长度,cmd: %s", json.dumps(cmd))
    stdout, return_code = util.run_cmd(cmd)
    if return_code != 0:
        logger.error("查询视频长度，异常")
        raise ValueError("查询视频长度，异常")
    return float(stdout)
