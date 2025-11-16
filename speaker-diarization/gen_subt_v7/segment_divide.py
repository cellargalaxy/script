import os
import tool_subt
import math
import util
from pydub import AudioSegment
import segment_detect_faster_whisper
import segment_detect_align_whisperx

logger = util.get_logger()

back_ms = 500


def box_segments(segments, start, end):
    segments = util.deepcopy_obj(segments)
    result = []
    for i, segment in enumerate(segments):
        if start < segment['end'] and segment['start'] < end:
            result.append(segment)
    tool_subt.check_discrete_segments(result)
    if len(result) > 0 and result[0]['start'] < start:
        result[0]['start'] = start
    if len(result) > 0 and end < result[-1]['end']:
        result[-1]['end'] = end
    tool_subt.check_discrete_segments(result)
    return result


def scrap_segments(segments, time):
    left = None
    middle = None
    right = None
    for i, segment in enumerate(segments):
        if segment['end'] < time:
            left = segment
        if segment['start'] <= time and time <= segment['end']:
            middle = segment
        if not right and time < segment['start']:
            right = segment
    return left, middle, right


def cut_segment(left_segment, right_segment, silences):
    left_side = (left_segment['end'] + left_segment['start']) / 2.0
    left_side = math.ceil(left_side)
    left_side = max(left_segment['end'] - back_ms, left_side)
    right_side = (right_segment['end'] + right_segment['start']) / 2.0
    right_side = math.floor(right_side)
    right_side = min(right_segment['start'] + back_ms, right_side)
    gaps = box_segments(silences, left_side, right_side)
    left_gap, middle_gap, right_gap = scrap_segments(gaps, right_segment['start'])
    left_cut = -1
    if left_gap:
        left_cut = (left_gap['end'] + left_gap['start']) / 2.0
        left_cut = math.floor(left_cut)
    middle_cut = -1
    if middle_gap:
        middle_cut = (middle_gap['end'] + middle_gap['start']) / 2.0
        middle_cut = math.floor(middle_cut)
    right_cut = -1
    if right_gap:
        right_cut = (right_gap['end'] + right_gap['start']) / 2.0
        right_cut = math.floor(right_cut)
    return left_cut, middle_cut, right_cut


def trim_silence(segment, silences):
    start_silence = None
    end_silence = None
    for i, silence in enumerate(silences):
        if silence['start'] <= segment['start'] and segment['start'] < silence['end']:
            start_silence = silence
        if silence['start'] < segment['end'] and segment['end'] <= silence['end']:
            end_silence = silence
    if start_silence:
        segment['start'] = start_silence['end']
    if end_silence:
        segment['end'] = end_silence['start']
    if segment['end'] <= segment['start']:
        return None
    return segment


def segment_divide(audio_path, part_detect_path, segment_detect_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    parts = util.read_file_to_obj(part_detect_path)

    silences = []
    for i, part in enumerate(parts):
        if part['vad_type'] != 'silence':
            continue
        silences.append(part)

    segments = util.read_file_to_obj(segment_detect_path)

    for i, segment in enumerate(segments):
        if i == 0:
            continue
        if segments[i]['start'] - segments[i - 1]['end'] >= 2 * back_ms:
            continue
        point = (segments[i]['start'] + segments[i - 1]['end']) / 2.0
        point = math.floor(point)
        segments[i - 1]['end'] = point
        segments[i]['start'] = point
    segments = tool_subt.fill_segments(segments, last_end=last_end, vad_type='silene')
    tool_subt.check_coherent_segments(segments)

    for i, segment in enumerate(segments):
        if i == 0:
            continue
        left_cut, middle_cut, right_cut = cut_segment(segments[i - 1], segments[i], silences)
        if middle_cut >= 0:
            segments[i - 1]['end'] = middle_cut
            segments[i]['start'] = middle_cut
            segments[i]['segment_divide_type'] = 'middle'
        elif left_cut >= 0 and right_cut < 0:
            segments[i - 1]['end'] = left_cut
            segments[i]['start'] = left_cut
            segments[i]['segment_divide_type'] = 'left_only'
        elif right_cut >= 0 and left_cut < 0:
            segments[i - 1]['end'] = right_cut
            segments[i]['start'] = right_cut
            segments[i]['segment_divide_type'] = 'right_only'
    tool_subt.check_coherent_segments(segments)

    results = []
    languages = []
    for i, segment in enumerate(segments):
        if i == 0:
            continue
        if segments[i].get('segment_divide_type', None):
            results.append(segments[i - 1])
        else:
            cut = audio[segments[i - 1]['start']:segments[i]['end']]
            segs, language = segment_detect_faster_whisper.transcribe(cut)
            languages.append(language)
            language = util.get_list_most(languages)
            segs, language = segment_detect_align_whisperx.transcribe(cut, segs, language)
            segs = tool_subt.shift_segments_time(segs, segments[i - 1]['start'])
            if not segs:
                segs = [segments[i]]
            results.extend(segs[:-1])
            segments[i] = segs[-1]
    results.append(segments[-1])
    segments = results
    for i, segment in enumerate(segments):
        if i == 0:
            continue
        if segments[i]['start'] - segments[i - 1]['end'] >= 2 * back_ms:
            continue
        point = (segments[i - 1]['end'] + segments[i]['start']) / 2.0
        point = math.floor(point)
        segments[i - 1]['end'] = point
        segments[i]['start'] = point
    segments = tool_subt.fill_segments(segments, last_end=last_end, vad_type='silene')
    tool_subt.check_coherent_segments(segments)

    for i, segment in enumerate(segments):
        if i == 0:
            continue
        if segments[i].get('segment_divide_type', None):
            continue
        left_cut, middle_cut, right_cut = cut_segment(segments[i - 1], segments[i], silences)
        if middle_cut >= 0:
            segments[i - 1]['end'] = middle_cut
            segments[i]['start'] = middle_cut
            segments[i]['segment_divide_type'] = 'middle'
        elif left_cut >= 0 and right_cut < 0:
            segments[i - 1]['end'] = left_cut
            segments[i]['start'] = left_cut
            segments[i]['segment_divide_type'] = 'left_only'
        elif right_cut >= 0 and left_cut < 0:
            segments[i - 1]['end'] = right_cut
            segments[i]['start'] = right_cut
            segments[i]['segment_divide_type'] = 'right_only'
        elif abs(left_cut - segments[i]['start']) < abs(right_cut - segments[i]['start']):
            segments[i - 1]['end'] = left_cut
            segments[i]['start'] = left_cut
            segments[i]['segment_divide_type'] = 'left_near'
        elif abs(right_cut - segments[i]['start']) < abs(left_cut - segments[i]['start']):
            segments[i - 1]['end'] = right_cut
            segments[i]['start'] = right_cut
            segments[i]['segment_divide_type'] = 'right_near'
    tool_subt.check_coherent_segments(segments)

    for i, segment in enumerate(segments):
        if segments[i].get('segment_divide_type', None):
            continue
        segments[i]['segment_divide_type'] = 'default'

    # results = []
    # for i, segment in enumerate(segments):
    #     if segment.get('vad_type', None) == 'silene':
    #         continue
    #     segment = trim_silence(segment, silences)
    #     if not segment:
    #         continue
    #     results.append(segment)
    # segments = results
    # if len(segments) > 0:
    #     if segments[0]['start'] < 1000:
    #         segments[0]['start'] = 0
    #     if last_end - segments[-1]['end'] < 1000:
    #         segments[-1]['end'] = last_end
    # for i, segment in enumerate(segments):
    #     if i == 0:
    #         continue
    #     if segments[i]['start'] - segments[i - 1]['end'] >= 1000:
    #         continue
    #     point = (segments[i - 1]['end'] + segments[i]['start']) / 2.0
    #     point = math.floor(point)
    #     segments[i - 1]['end'] = point
    #     segments[i]['start'] = point
    # segments = tool_subt.fill_segments(segments, last_end=last_end, vad_type='silene')
    # tool_subt.check_coherent_segments(segments)

    results = []
    for i, segment in enumerate(segments):
        if segment.get('vad_type', None) == 'silene':
            continue
        results.append(segment)
    segments = results

    segments = tool_subt.fix_overlap_segments(segments)
    segments = tool_subt.unit_segments(segments, 'vad_type')
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
    segment_detect_faster_whisper.exec_gc()
    segment_detect_align_whisperx.exec_gc()
