import ffmpeg_util
import ffprobe_util
import demucs_demo
import json
import util
import whisper_timestamped_demo
import pyannote_audio_split_demo
import join_sub
import pretreatment

logger = util.get_logger()

manager = {
    "video_path": "demo.mkv",
    "output_dir": "output/demo",
    "audio_path": 'output/demo/extract_audio/wav.wav',
    "demucs_audio_path": 'output/demo/demucs/htdemucs/wav/vocals.wav',
    "demucs_video_path": 'output/demo/demucs/mkv.mkv',
    "auth_token": '',
    "min_silene_duration": 2,
    "edge_duration": 1,
    "speech_duration": 30,
    "split_video_dir": 'output/demo/split_video',
    "subtitle_save_dir": 'output/demo/subtitle',
}

util.print_device_info()

pretreatment.init_param_by_manager(manager)
pretreatment.extract_audio_by_manager(manager)
# 可以考虑加个降噪
# demucs_demo.demucs_and_join_by_manager(manager)
# pyannote_audio_split_demo.split_video_by_manager(manager)
# whisper_timestamped_demo.whisper_timestamped_by_manager(manager)
# join_sub.join_sub_and_save_by_manager(manager)

logger.info("manager: %s", json.dumps(manager))
