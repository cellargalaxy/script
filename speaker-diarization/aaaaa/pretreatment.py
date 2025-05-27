import ffprobe_util
import json
import util
import time
import ffmpeg_util
import os

logger = util.get_logger()


def init_param_by_manager(manager):
    video_path = manager.get('video_path')
    file_name = util.get_file_name(video_path)
    output_dir = os.path.join('output', file_name)
    manager['output_dir'] = output_dir


def extract_audio(video_path, audio_path):
    audio_tracks = ffprobe_util.get_audio_track_info(video_path)
    logger.info("提取音轨，音轨列表: %s", json.dumps(audio_tracks))
    if len(audio_tracks) == 0:
        logger.error("提取音轨，音轨为空")
        raise ValueError("提取音轨，音轨为空")
    audio_track_index = 0
    if len(audio_tracks) > 1:
        for track in audio_tracks:
            logger.info("提取音轨，音轨, %s:%s", track.get('index', 0) - 1, track.get('tags', {}).get('track', '未知'))
        time.sleep(0.5)
        audio_track_index = int(input("输入音轨编号: "))

    ffmpeg_util.extract_audio_track(video_path, audio_track_index, audio_path)


def extract_audio_by_manager(manager):
    video_path = manager.get('video_path')
    audio_path = os.path.join(manager.get('output_dir'), "extract_audio/wav.wav")
    extract_audio(video_path, audio_path)
    manager['audio_path'] = audio_path
