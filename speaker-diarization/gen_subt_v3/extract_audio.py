import util_ffprobe
import json
import util
import time
import util_ffmpeg
import os

logger = util.get_logger()


def extract_audio(video_path, audio_path, audio_track_index=0):
    if util.path_exist(audio_path):
        return
    audio_tracks = util_ffprobe.get_audio_track_info(video_path)
    logger.info("提取音轨，音轨列表: %s", json.dumps(audio_tracks))
    if len(audio_tracks) == 0:
        logger.error("提取音轨，音轨为空")
        raise ValueError("提取音轨，音轨为空")
    track_index = 0
    while len(audio_tracks) > 1:
        for track in audio_tracks:
            logger.info("提取音轨，音轨, %s:%s", track.get('index', 0) - 1,
                        track.get('tags', {}).get('language', '未知'))
        time.sleep(0.5)
        track_index = int(util.input_timeout("输入音轨编号: ", 60, audio_track_index))
        if track_index < 0 or len(audio_tracks) <= track_index:
            logger.info("提取音轨，音轨选择错误")
            continue
        break
    util_ffmpeg.extract_audio_track(video_path, track_index, audio_path)


def exec(manager):
    logger.info("extract_audio,enter: %s", json.dumps(manager))
    video_path = manager.get('video_path')
    audio_track_index = manager.get('audio_track_index', 0)
    audio_path = os.path.join(manager.get('output_dir'), "extract_audio", "wav.wav")
    extract_audio(video_path, audio_path, audio_track_index=audio_track_index)
    manager['audio_path'] = audio_path
    logger.info("extract_audio,leave: %s", json.dumps(manager))
