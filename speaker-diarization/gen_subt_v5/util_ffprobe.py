import json
import util

logger = util.get_logger()


def get_audio_track_info(video_path):
    """
    :param video_path:
    :return: [{"index": 1, "tags": {"language": "jpn"}}, {"index": 2, "tags": {"language": "eng"}}]
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream',
        '-of', 'json',
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
