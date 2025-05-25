import ffmpeg_util
import ffprobe_util
import demucs_demo
import json
import util
import whisper_timestamped_demo
import pyannote_audio_split_demo

logger = util.get_logger()

manager = {
    "video_path": "demo.mkv",
    "output_dir": "output/demo",
    # "audio_track_index": 1,
    # "audio_path": 'output/demo/extract_audio_track/wav.wav',
    # "device": 'cpu',
    "demucs_audio_path": 'output/demo/demucs/htdemucs/wav/vocals.wav',
    "demucs_video_path": 'output/demo/demucs/mkv.mkv',
    # "auth_token": '',
    "min_silene_duration": 2,
    "edge_duration": 1,
    # "split_video_dir": 'output/demo/split_video',
}

util.print_device_info()
util.set_device_type_to_manager(manager)

# ffprobe_util.choice_audio_track_info_in_terminal_by_manager(manager)
# ffmpeg_util.extract_audio_track_by_manager(manager)
# 可以考虑加个降噪
# demucs_demo.demucs_and_join_by_manager(manager)
pyannote_audio_split_demo.split_video_by_manager(manager)
whisper_timestamped_demo.whisper_timestamped_by_manager(manager)

logger.info("manager: %s", json.dumps(manager))
