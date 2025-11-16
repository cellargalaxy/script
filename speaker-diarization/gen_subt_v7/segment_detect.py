import util
import os
import segment_detect_faster_whisper
from pydub import AudioSegment
import tool_subt
import segment_detect_align_whisperx

logger = util.get_logger()


def segment_detect(audio_path, part_divide_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_detect.json')
    srt_path = os.path.join(output_dir, 'segment_detect.srt')
    if util.path_exist(json_path):
        return json_path

    logger.info("字幕生成: %s", audio_path)

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    segments = util.read_file_to_obj(part_divide_path)

    results = []
    languages = []
    for i, segment in enumerate(segments):
        if segment['vad_type'] != 'speech':
            continue
        cut = audio[segment['start']:segment['end']]
        segs, language = segment_detect_faster_whisper.transcribe(cut)
        segs = tool_subt.shift_segments_time(segs, segment['start'])
        results.extend(segs)
        languages.append(language)
    language = util.get_list_most(languages)
    for i, segment in enumerate(segments):
        segments[i]['language'] = language

    results, language = segment_detect_align_whisperx.transcribe(audio, results, language)

    results = tool_subt.fix_overlap_segments(results)
    results = tool_subt.clipp_segments(results, last_end)
    results = tool_subt.init_segments(results)
    tool_subt.check_discrete_segments(results)

    util.save_as_json(results, json_path)
    tool_subt.save_segments_as_srt(results, srt_path)
    return json_path


def exec(manager):
    logger.info("segment_detect,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    part_divide_path = manager.get('part_divide_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_detect")
    json_path = segment_detect(audio_path, part_divide_path, output_dir)
    manager['segment_detect_path'] = json_path
    logger.info("segment_detect,leave: %s", util.json_dumps(manager))
    segment_detect_faster_whisper.exec_gc()
    segment_detect_align_whisperx.exec_gc()
