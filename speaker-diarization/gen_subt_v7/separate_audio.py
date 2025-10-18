import util_ffprobe
import util
import time
import util_ffmpeg
import os

logger = util.get_logger()


def separate_audio(video_path, output_dir):
    audio_path = os.path.join(output_dir, "wav.wav")
    if util.path_exist(audio_path):
        return audio_path

    audio_tracks = util_ffprobe.get_audio_track_info(video_path)
    if len(audio_tracks) == 0:
        logger.error("分离音轨，音轨为空")
        raise ValueError("分离音轨，音轨为空")
    track_index = 0
    while len(audio_tracks) > 1:
        for track in audio_tracks:
            index = track.get('index', 0) - 1
            language = track.get('tags', {}).get('language', '未知')
            logger.info("分离音轨，音轨, %s:%s", index, language)
        time.sleep(0.5)
        track_index = int(util.input_timeout("输入音轨编号: ", 5, track_index))
        if track_index < 0 or len(audio_tracks) <= track_index:
            logger.error("分离音轨，音轨选择错误")
            continue
        break
    util_ffmpeg.extract_audio_track(video_path, track_index, audio_path)
    return audio_path


def exec(manager):
    logger.info("separate_audio,enter: %s", util.json_dumps(manager))
    video_path = manager.get('video_path')
    output_dir = os.path.join(manager.get('output_dir'), "separate_audio")
    audio_path = separate_audio(video_path, output_dir)
    manager['audio_path'] = audio_path
    logger.info("separate_audio,leave: %s", util.json_dumps(manager))
