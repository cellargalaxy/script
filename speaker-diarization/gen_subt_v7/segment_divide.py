import util
import os
import tool_subt
from pydub import AudioSegment
import part_detect_vad

logger = util.get_logger()


def find_first_start(segments, start, end):
    for i, segment in enumerate(segments):
        start_overlap = max(start, segment['start'])
        end_overlap = min(end, segment['end'])
        if start_overlap <= end_overlap:
            return start_overlap
    return None


def find_first_end(segments, start, end):
    for i, segment in enumerate(segments):
        start_overlap = max(start, segment['start'])
        end_overlap = min(end, segment['end'])
        if start_overlap <= end_overlap:
            return end_overlap
    return None


def segment_divide(audio_path, part_detect_path, segment_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    parts = util.read_file_to_obj(part_detect_path)
    segments = util.read_file_to_obj(segment_detect_path)

    silences = []
    for i, part in enumerate(parts):
        if part['vad_type'] != 'silence':
            continue
        silences.append(part)

    for i, segment in enumerate(segments):
        if i == 0:
            continue

        back = min(500, int(segments[i]['duration'] / 2))

        pre_end = segments[i - 1]['end']
        start = segments[i]['start'] + back
        find_start = pre_end
        find_end = min(pre_end + 1000, start)
        ms = find_first_start(silences, find_start, find_end)
        if ms:
            segments[i - 1]['end'] = ms

        pre_end = segments[i - 1]['end']
        start = segments[i]['start'] + back
        find_start = max(start - 1000 - back, pre_end)
        find_end = start
        ms = find_first_end(silences, find_start, find_end)
        if ms:
            segments[i]['start'] = ms
        if segments[i]['start'] < segments[i - 1]['end']:
            segments[i]['start'] = segments[i - 1]['end']

    audio = AudioSegment.from_wav(audio_path)
    for i, segment in enumerate(segments):
        if segments[i]['end'] - segments[i]['start'] < 200:
            continue
        cut = audio[segments[i]['start']:segments[i]['end']]
        segs = part_detect_vad.part_detect_by_data(cut)
        if len(segs) < 2:
            continue
        if segs[0]['vad_type'] == 'silence':
            segments[i]['start'] = segments[i]['start'] + max(segs[0]['duration'] - 500, 0)
        if segs[-1]['vad_type'] == 'silence':
            segments[i]['end'] = segments[i]['end'] - max(segs[-1]['duration'] - 500, 0)

    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.init_segments(segments)
    tool_subt.check_discrete_segments(segments)

    util.save_as_json(segments, json_path)
    tool_subt.save_segments_as_srt(segments, srt_path)
    return json_path


def exec(manager):
    logger.info("segment_divide,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    part_detect_path = manager.get('part_detect_path')
    segment_detect_path = manager.get('segment_detect_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_divide")
    json_path = segment_divide(audio_path, part_detect_path, segment_detect_path, output_dir)
    manager['segment_divide_path'] = json_path
    logger.info("segment_divide,leave: %s", util.json_dumps(manager))
    util.exec_gc()
