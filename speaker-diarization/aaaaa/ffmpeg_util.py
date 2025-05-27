import json
import util
import os

logger = util.get_logger()


def extract_audio_track(video_path, index, audio_path):
    util.mkdir(audio_path)
    cmd = [
        'bin/ffmpeg',
        '-i', video_path,
        '-map', "0:a:{}".format(index),  # 在第0个输入文件中选择第index个音轨
        '-ac', '1',  # 将音频转换为单声道
        '-y',
        audio_path
    ]
    logger.info("提取视频文件音轨,cmd: %s", json.dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("提取视频文件音轨，异常")
        raise ValueError("提取视频文件音轨，异常")


def join_video_and_audio(video_path, audio_path, output_path):
    util.mkdir(output_path)
    cmd = [
        'bin/ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        '-y',
        output_path
    ]
    logger.info("合并视频音频,cmd: %s", json.dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("合并视频音频，异常")
        raise ValueError("合并视频音频，异常")


def cut_video(video_path, start, end, output_path):
    util.mkdir(output_path)
    cmd = [
        'bin/ffmpeg',
        '-ss', str(start),
        '-to', str(end),
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '28',
        '-vf', 'scale=640:-1',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-y',
        output_path
    ]
    logger.info("剪切视频,cmd: %s", json.dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("剪切视频，异常")
        raise ValueError("剪切视频，异常")
