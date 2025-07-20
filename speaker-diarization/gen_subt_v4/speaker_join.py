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
    audio_data.export(output_path, format='wav')


def speaker_join(speaker_divide_path, output_dir):
    if util.path_exist(output_dir):
        return

    content = util.read_file(speaker_divide_path)
    segments = json.loads(content)

    speaker_map = {}
    for i, segment in enumerate(segments):
        speakers = speaker_map.get(segment['speaker'], [])
        speakers.append(segment)
        speaker_map[segment['speaker']] = speakers

    for speaker in speaker_map:
        if speaker == 'unknown':
            continue
        segments = speaker_map[speaker]
        if len(segments) < 2:
            continue
        audio_paths = []
        speaker_dir = os.path.join(output_dir, speaker)
        for i, segment in enumerate(segments):
            audio_paths.append(segment['wav_path'])
            util.copy_file(segment['wav_path'], os.path.join(speaker_dir, f"{segment['file_name']}.wav"))
        audio_path = os.path.join(output_dir, f"{speaker}.wav")
        join_audio(audio_paths, audio_path)


def exec(manager):
    logger.info("speaker_join,enter: %s", json.dumps(manager))
    speaker_divide_path = manager.get('speaker_divide_path')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_join")
    speaker_join(speaker_divide_path, output_dir)
    manager['speaker_join_path'] = output_dir
    logger.info("speaker_join,leave: %s", json.dumps(manager))
    util.exec_gc()
