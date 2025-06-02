import json
import util
import pre_treatment
import noise_reduction_demucs
import audio_split_silero_vad
import speaker_diarization_pyannote

logger = util.get_logger()

manager = {
    "video_path": "../demo.mkv",
    "audio_track_index": 0,
    "auth_token": "",

    # "output_dir": "output/demo",
    # "audio_path": 'output/demo/extract_audio/wav.wav',
    # "noise_reduction_audio_path": 'output/demo/noise_reduction/htdemucs/wav/vocals.wav',
    # "audio_combine_wav_path": "output/demo/audio_split/combine.wav",
    # "audio_combine_json_path": "output/demo/audio_split/combine.json",
    # "audio_segment_dir": "output/demo/audio_split/segment",
    # "audio_class_dir": "output/demo/audio_class",
}

util.print_device_info()
pre_treatment.init_by_manager(manager)

pre_treatment.extract_audio_by_manager(manager)
noise_reduction_demucs.noise_reduction_by_manager(manager)
audio_split_silero_vad.split_video_by_manager(manager)
speaker_diarization_pyannote.audio_class_by_manager(manager)

logger.info("manager: %s", json.dumps(manager))
