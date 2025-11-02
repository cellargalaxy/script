import util

logger = util.get_logger()


def extract_audio_track(video_path, index, audio_path):
    util.mkdir(audio_path)
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-map', f"0:a:{index}",  # 在第0个输入文件中选择第index个音轨
        '-ac', '2',
        '-c:a', 'pcm_s16le',
        '-y',
        audio_path
    ]
    logger.info("提取视频文件音轨,cmd: %s", util.json_dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("提取视频文件音轨，异常")
        raise ValueError("提取视频文件音轨，异常")


def extract_simple_audio(input_path, output_path):
    util.mkdir(output_path)
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-ac', '1',
        '-ar', '16000',
        '-y',
        output_path
    ]
    logger.info("合并音频声道,cmd: %s", util.json_dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("合并音频声道，异常")
        raise ValueError("合并音频声道，异常")
