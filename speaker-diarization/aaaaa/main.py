import ffmpeg_util
import ffprobe_util
import demucs_demo
import json
import util
import split_demo

logger = util.get_logger()

manager = {
    "video_path": "demo.mkv",
    "output_dir": "output/demo",
    # "audio_track_index": 1,
    # "audio_path": 'output/demo/extract_audio_track/wav.wav',
    # "device": 'cpu',
    # "demucs_audio_path": 'output/demo/demucs/htdemucs/wav/vocals.wav',
    # "demucs_video_path": 'output/demo/demucs/mkv.mkv',
    "split_video_window": 200,
    "split_video_overlap": 20,
}

util.print_device_info()
util.set_device_type_to_manager(manager)

ffprobe_util.choice_audio_track_info_in_terminal_by_manager(manager)
ffmpeg_util.extract_audio_track_by_manager(manager)
demucs_demo.demucs_and_join_by_manager(manager)
split_demo.split_video_by_manager(manager)

logger.info("manager: %s", json.dumps(manager))
