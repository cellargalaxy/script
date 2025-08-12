import util
import json
import util_subt
from pydub import AudioSegment
import os

logger = util.get_logger()


def part_divide(audio_path, part_detect_path, output_dir,
                min_speech_duration_ms=300,
                min_silence_duration_ms=300,
                part_speech_duration_ms=1000 * 15):
    json_path = os.path.join(output_dir, 'part_divide.json')
    srt_path = os.path.join(output_dir, 'part_divide.srt')
    if util.path_exist(json_path):
        return json_path

    audio = AudioSegment.from_wav(audio_path)

    segments = util.read_file_to_obj(part_detect_path)

    for i, segment in enumerate(segments):
        if segments[i]['end'] - segments[i]['start'] < min_speech_duration_ms:
            segments[i]['too_mini_speech'] = True
    segments = util_subt.gradual_segments(segments, gradual_duration_ms=min_silence_duration_ms, audio_data=audio)
    for i, segment in enumerate(segments):
        if segments[i].get('too_mini_speech', False):
            segments[i]['vad_type'] = 'silence'
            del segments[i]['too_mini_speech']
    segments = util_subt.unit_segments(segments, 'vad_type')
    # util.save_as_json(segments, os.path.join(output_dir, 'gradual.json'))
    # util_subt.save_segments_as_srt(segments, os.path.join(output_dir, 'gradual.srt'), skip_silence=True)

    parts = []
    for i, segment in enumerate(segments):
        if len(parts) == 0:
            parts.append({"start": 0, "end": 0})
            continue
        if segment['end'] - parts[-1]['start'] < part_speech_duration_ms:
            continue
        if segment['vad_type'] != 'silence':
            continue
        if segment['vad_type'] == 'silence':
            parts[-1]['end'] = segments[i]['start']
            parts.append(segments[i])
            parts.append({"start": segments[i]['end'], "end": 0})
        else:
            parts[-1]['end'] = segments[i]['end']
            parts.append({"start": segments[i]['end'], "end": 0})
    if len(parts) > 1:
        parts.pop()
    parts[-1]['end'] = segments[-1]['end']
    parts[-1]['vad_type'] = 'silence'
    for i, segment in enumerate(segments):
        if segment['start'] < parts[-1]['start']:
            continue
        if segment['vad_type'] == 'speech':
            parts[-1]['vad_type'] = 'speech'
            break

    segments = []
    for i, segment in enumerate(parts):
        if segment['start'] == segment['end']:
            continue
        if segment.get('vad_type', '') not in ('silence', 'speech'):
            segment['vad_type'] = 'speech'
        segments.append(segment)

    for i, segment in enumerate(segments):
        segments[i]['file_name'] = f"{i:05d}_{segments[i]['vad_type']}"

    util_subt.check_coherent_segments(segments)
    util.save_as_json(segments, json_path)
    util_subt.save_segments_as_srt(segments, srt_path, skip_silence=True)
    return json_path


def exec(manager):
    logger.info("part_divide,enter: %s", json.dumps(manager))
    audio_path = manager.get('audio_path')
    part_detect_path = manager.get('part_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "part_divide")
    part_divide_path = part_divide(audio_path, part_detect_path, output_dir)
    manager['part_divide_path'] = part_divide_path
    logger.info("part_divide,leave: %s", json.dumps(manager))
    util.exec_gc()
