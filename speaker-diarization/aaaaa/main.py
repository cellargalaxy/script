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

# whisperx与whisper_timestamped对比
# 耗时：whisperx比whisper_timestamped快一倍
# 字符时间识别准确性：whisper_timestamped比whisperx明显好，whisperx是不是有些字会卡住不动
# 句子时间识别准确性：差不多，whisper_timestamped比whisperx一些些，但并不多
# 句子内容识别准确性：whisper_timestamped比whisperx好一些，但whisperx也勉强够用
# 句子断句：whisper_timestamped会把同一个人的短句连在一起，而whisperx会将句子切割的更碎
# 综上所述：如果只是以句子颗粒度切割语音，whisper_timestamped与whisperx基本差不多，但鉴于whisperx比whisper_timestamped快一倍，选择whisperx
# transcribe_sub_whisper_timestamped.transcribe_and_save_sub_by_manager(manager)
transcribe_sub_whisperx.transcribe_and_save_sub_by_manager(manager)

# join_sub.join_sub_and_save_by_manager(manager)

logger.info("manager: %s", json.dumps(manager))
