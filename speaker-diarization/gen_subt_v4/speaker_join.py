import os
import util
import json
from pydub import AudioSegment

logger = util.get_logger()


def join_audio(audio_paths, output_path, silene_duration_ms=350):
    blank_data = AudioSegment.silent(duration=silene_duration_ms)
    audio_data = AudioSegment.silent(duration=silene_duration_ms)
    for audio_path in audio_paths:
        audio = AudioSegment.from_wav(audio_path)
        audio_data = audio_data + audio + blank_data
    util.mkdir(output_path)
    audio_data.export(output_path, format="wav")


def speaker_join(speaker_divide_dir, output_dir):
    if util.path_exist(output_dir):
        return

    speaker_divide_dir = os.path.join(speaker_divide_dir, "split")
    speaker_dirs = util.listdir(speaker_divide_dir)
    for speaker_dir in speaker_dirs:
        audio_paths = util.listdir(os.path.join(speaker_divide_dir, speaker_dir))
        for i, segment in enumerate(audio_paths):
            audio_paths[i] = os.path.join(speaker_divide_dir, speaker_dir, audio_paths[i])
        audio_path = os.path.join(output_dir, f"{speaker_dir}.wav")
        join_audio(audio_paths, audio_path)


def exec(manager):
    logger.info("speaker_join,enter: %s", json.dumps(manager))
    speaker_divide_dir = manager.get('speaker_divide_dir')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_join")
    speaker_join(speaker_divide_dir, output_dir)
    manager['speaker_join_path'] = output_dir
    logger.info("speaker_join,leave: %s", json.dumps(manager))
    util.exec_gc()
