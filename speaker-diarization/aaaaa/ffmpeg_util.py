import json
import util
import os

logger = util.get_logger()


def extract_audio_track(video_path, index, audio_path):
    util.mkdir(audio_path)
    cmd = [
        'bin/ffmpeg',
        '-i', video_path,
        '-map', "0:a:{}".format(index),  # 从第一个输入文件（0）中选择第 index 个音轨（音频流）
        '-ac', '1',  # 将音频转换为单声道
        '-y',  # 强制覆盖输出文件
        audio_path
    ]
    logger.info("提取视频文件音轨,cmd: %s", json.dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("提取视频文件音轨，异常")
        raise ValueError("提取视频文件音轨，异常")


def extract_audio_track_by_manager(manager):
    video_path = manager.get('video_path')
    index = manager.get('audio_track_index')
    audio_path = os.path.join(manager.get('output_dir'), "extract_audio_track", "wav.wav")
    manager['audio_path'] = audio_path
    extract_audio_track(video_path, index, audio_path)


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
        '-i', video_path,
        '-ss', str(start),
        '-to', str(end),
        '-c', 'copy',
        '-y',
        output_path
    ]
    logger.info("剪切视频,cmd: %s", json.dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("剪切视频，异常")
        raise ValueError("剪切视频，异常")
