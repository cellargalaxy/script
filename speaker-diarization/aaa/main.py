import json
import util
import pre_treatment
import noise_reduction_demucs

logger = util.get_logger()

manager = {
    "video_path": "../demo.mkv",

    # "output_dir": "output/demo",
    # "audio_path": 'output/demo/extract_audio/wav.wav',
    # "noise_reduction_audio_path": 'output/demo/noise_reduction/htdemucs/wav/vocals.wav',
}

util.print_device_info()
pre_treatment.init_param_by_manager(manager)

pre_treatment.extract_audio_by_manager(manager)
noise_reduction_demucs.noise_reduction_by_manager(manager)

logger.info("manager: %s", json.dumps(manager))
