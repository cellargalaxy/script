import json
import time
import util
import pre_treatment
import noise_reduction_demucs
import audio_split_pyannote
import transcribe_sub_whisper_timestamped
import transcribe_sub_whisperx

logger = util.get_logger()

manager = {
    "video_path": "demo.mkv",

    # "output_dir": "output/demo",
    # "audio_path": 'output/demo/extract_audio/wav.wav',
    # "noise_reduction_audio_path": 'output/demo/noise_reduction/htdemucs/wav/vocals.wav',

    "auth_token": "",
    "min_silene_duration": 2 * 1000,
    "edge_duration": 1 * 1000,
    "speech_duration": 30 * 1000,

    # "split_video_dir": 'output/demo/split_video',
    # "transcribe_sub_dir": 'output/demo/transcribe_sub',
}

util.print_device_info()
pre_treatment.init_param_by_manager(manager)

pre_treatment.extract_audio_by_manager(manager)
noise_reduction_demucs.noise_reduction_by_manager(manager)
audio_split_pyannote.split_video_by_manager(manager)
transcribe_sub_whisperx.transcribe_and_save_sub_by_manager(manager)
time.sleep(10)
transcribe_sub_whisper_timestamped.transcribe_and_save_sub_by_manager(manager)
# join_sub.join_sub_and_save_by_manager(manager)

logger.info("manager: %s", json.dumps(manager))
