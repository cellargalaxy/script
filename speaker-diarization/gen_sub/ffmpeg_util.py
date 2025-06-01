import json
import util

logger = util.get_logger()


def extract_audio_track(video_path, index, audio_path):
    util.mkdir(audio_path)
    cmd = [
        'bin/ffmpeg',
        '-i', video_path,
        '-map', "0:a:{}".format(index),  # 在第0个输入文件中选择第index个音轨
        '-ac', '1',  # 将音频转换为单声道
        '-ar', '16000',  # 16kHz
        '-y',
        audio_path
    ]
    logger.info("提取视频文件音轨,cmd: %s", json.dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("提取视频文件音轨，异常")
        raise ValueError("提取视频文件音轨，异常")
