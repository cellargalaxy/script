import json
import util
import os
import ffmpeg_util

logger = util.get_logger()


def demucs(audio_path, device, output_dir): #noise_reduction_demucs
    util.mkdir(audio_path)
    cmd = [
        'demucs',
        audio_path,
        '-v',
        '-d', device,
        '-o', output_dir,
    ]
    logger.info("去除音频音乐,cmd: %s", json.dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("去除音频音乐，异常")
        raise ValueError("去除音频音乐，异常")
    output_path = os.path.join(output_dir, "htdemucs/wav/vocals.wav")
    return output_path


def demucs_by_manager(manager):
    audio_path = manager.get('audio_path')
    device = manager.get('device')
    output_dir = os.path.join(manager.get('output_dir'), "demucs")
    output_path = demucs(audio_path, device, output_dir)
    manager['demucs_audio_path'] = output_path


def demucs_and_join(video_path, audio_path, device, output_dir):
    output_path = demucs(audio_path, device, output_dir)
    ext = util.get_file_ext(video_path)
    video_demucs_path = os.path.join(output_dir, "{}.{}".format(ext, ext))
    ffmpeg_util.join_video_and_audio(video_path, output_path, video_demucs_path)
    return video_demucs_path


def demucs_and_join_by_manager(manager):
    audio_path = manager.get('audio_path')
    device = manager.get('device')
    output_dir = os.path.join(manager.get('output_dir'), "demucs")
    demucs_audio_path = demucs(audio_path, device, output_dir)
    manager['demucs_audio_path'] = demucs_audio_path
