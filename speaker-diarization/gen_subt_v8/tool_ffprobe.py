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
    stdout, return_code = util.run_cmd(cmd)
    if return_code != 0:
        logger.error("查询文件音轨，异常")
        raise ValueError("查询文件音轨，异常")
    info = util.json_loads(stdout)
    logger.info("查询文件音轨: %s", util.json_dumps(info))
    audio_streams = info.get('streams', [])
    return audio_streams
