import util
import os
from pydub import AudioSegment

logger = util.get_logger()


def part_silence(audio_path, part_detect_path, output_path):
    if util.path_exist(output_path):
        return output_path
    audio = AudioSegment.from_wav(audio_path)
    segments = util.read_file_to_obj(part_detect_path)
    for i, segment in enumerate(segments):
        if segment['vad_type'] != 'speech':
            start = segment['start']
            end = segment['end']
            audio = (
                    audio[:start] +
                    audio[start:end].apply_gain(-120) +
                    audio[end:]
            )
    util.mkdir(output_path)
    audio.export(output_path, format="wav")
    return output_path


def exec(manager):
    logger.info("part_silence,enter: %s", util.json_dumps(manager))
    part_detect_path = manager.get('part_detect_path')
    audio_path = manager.get('audio_path')
    split_audio_path = manager.get('split_audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "part_silence")
    audio_path = part_silence(audio_path, part_detect_path, os.path.join(output_dir, "audio.wav"))
    split_audio_path = part_silence(split_audio_path, part_detect_path, os.path.join(output_dir, "split_audio.wav"))
    manager['audio_path'] = audio_path
    manager['split_audio_path'] = split_audio_path
    logger.info("part_silence,leave: %s", util.json_dumps(manager))
    util.exec_gc()
