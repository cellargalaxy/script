import os
import util
import json

logger = util.get_logger()


def speaker_segment(segment_paths, output_dir, min_speech_duration_ms=1000):
    json_path = os.path.join(output_dir, 'speaker_segment.json')
    if util.path_exist(json_path):
        return json_path

    speaks = []
    for i, segment_path in enumerate(segment_paths):
        segments = util.read_file_to_obj(segment_path)
        speeches = []
        for j, segment in enumerate(segments):
            if segment['vad_type'] != 'speech':
                continue
            if segment['end'] - segment['start'] < min_speech_duration_ms:
                continue
            speeches.append(segment)
        for j, segment in enumerate(speeches):
            speak = {
                'file_name': f"speaker{i:03d}{j:03d}",
                'segments': [segment],
            }
            speaks.append(speak)

    util.save_as_json(speaks, json_path)
    return json_path


def exec(manager):
    logger.info("speaker_segment,enter: %s", json.dumps(manager))
    segment_paths = manager.get('segment_paths', [])
    segment_path = manager.get('segment_split_path', None)
    if segment_path:
        segment_paths.append(segment_path)
    output_dir = os.path.join(manager.get('output_dir'), "speaker_segment")
    speaker_segment_path = speaker_segment(segment_paths, output_dir)
    manager['speaker_segment_path'] = speaker_segment_path
    logger.info("speaker_segment,leave: %s", json.dumps(manager))
    util.exec_gc()
