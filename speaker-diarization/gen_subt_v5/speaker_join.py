import os
import util
import json
from pydub import AudioSegment

logger = util.get_logger()


def join_audio(speak, silence_duration_ms=100):
    blank_data = AudioSegment.silent(duration=silence_duration_ms)
    audio_data = AudioSegment.silent(duration=silence_duration_ms)
    for i, segment in enumerate(speak['segments']):
        audio = AudioSegment.from_wav(segment['wav_path'])
        audio_data = audio_data + audio + blank_data
    util.mkdir(speak['wav_path'])
    audio_data.export(speak['wav_path'], format='wav')


def speaker_join(speak_path, output_dir):
    json_path = os.path.join(output_dir, 'speaker_join.json')
    if util.path_exist(json_path):
        return json_path

    content = util.read_file(speak_path)
    speaks = json.loads(content)

    for i, speak in enumerate(speaks):
        speaks[i]['wav_path'] = os.path.join(output_dir, 'speaker', f"{speaks[i]['file_name']}.wav")

    for i, speak in enumerate(speaks):
        if speak['file_name'] == 'unknown':
            continue
        join_audio(speak)
        for j, segment in enumerate(speak['segments']):
            copy_path = os.path.join(output_dir, 'speaker', speaks[i]['file_name'], f"{segment['file_name']}.wav")
            util.copy_file(segment['wav_path'], copy_path)

    util.save_as_json(speaks, json_path)
    return json_path


def exec(manager):
    logger.info("speaker_join,enter: %s", json.dumps(manager))
    speaker_segment_path = manager.get('speaker_segment_path')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_join")
    speaker_join_path = speaker_join(speaker_segment_path, output_dir)
    manager['speaker_join_path'] = speaker_join_path
    logger.info("speaker_join,leave: %s", json.dumps(manager))
    util.exec_gc()
