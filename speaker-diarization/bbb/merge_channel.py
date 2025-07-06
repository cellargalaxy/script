import util_ffprobe
import json
import util
import time
import util_ffmpeg
import os

logger = util.get_logger()


def merge_channel(input_path, output_path):
    if util.path_exist(output_path):
        return
    util_ffmpeg.merge_audio_channel(input_path, output_path)


def exec(manager):
    logger.info("merge_channel,enter: %s", json.dumps(manager))
    input_path = manager.get('noise_reduction_path')
    output_path = os.path.join(manager.get('output_dir'), "merge_channel", "wav.wav")
    merge_channel(input_path, output_path)
    manager['merge_channel_path'] = output_path
    logger.info("merge_channel,leave: %s", json.dumps(manager))
