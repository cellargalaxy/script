import ffprobe_util
import json
import util
import time
import ffmpeg_util
import os

logger = util.get_logger()


def init_shell():
    cmd = [
        'sh', 'init.sh',
    ]
    logger.info("初始化,cmd: %s", json.dumps(cmd))
    stdout, return_code = util.run_cmd(cmd)
    if return_code != 0:
        logger.error("初始化，异常")
        raise ValueError("初始化，异常")


def init_by_manager(manager):
    init_shell()
    video_path = manager.get('video_path')
    file_name = util.get_file_name(video_path)
    output_dir = os.path.join('output', file_name)
    manager['output_dir'] = output_dir


def extract_audio(video_path, audio_path, audio_track_index=0):
    audio_tracks = ffprobe_util.get_audio_track_info(video_path)
    logger.info("提取音轨，音轨列表: %s", json.dumps(audio_tracks))
    if len(audio_tracks) == 0:
        logger.error("提取音轨，音轨为空")
        raise ValueError("提取音轨，音轨为空")
    while len(audio_tracks) > 1:
        for track in audio_tracks:
            logger.info("提取音轨，音轨, %s:%s", track.get('index', 0) - 1,
                        track.get('tags', {}).get('language', '未知'))
        time.sleep(0.5)
        audio_track_index = int(util.input_timeout("输入音轨编号: ", 60, audio_track_index))
        if audio_track_index < 0 or len(audio_tracks) <= audio_track_index:
            logger.info("提取音轨，音轨选择错误")
            continue
        break
    ffmpeg_util.extract_audio_track(video_path, audio_track_index, audio_path)


def extract_audio_by_manager(manager):
    video_path = manager.get('video_path')
    audio_track_index = manager.get('audio_track_index', 0)
    audio_path = os.path.join(manager.get('output_dir'), "extract_audio/wav.wav")
    extract_audio(video_path, audio_path, audio_track_index=audio_track_index)
    manager['audio_path'] = audio_path
