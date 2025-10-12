import util
import os
from pydub import AudioSegment
import util_subt

logger = util.get_logger()


def segment_split(audio_path, segment_divide_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_split.json')
    srt_path = os.path.join(output_dir, 'segment_split.srt')
    if util.path_exist(json_path):
        return json_path

    segments = util.read_file_to_obj(segment_divide_path)

    split_dir = os.path.join(output_dir, 'split')
    util.mkdir(split_dir)
    for i, segment in enumerate(segments):
        segments[i]['wav_path'] = os.path.join(split_dir, f"{segments[i]['file_name']}.wav")

    audio = AudioSegment.from_wav(audio_path)
    for i, segment in enumerate(segments):
        cut_path = segment['wav_path']
        cut = audio[segment['start']:segment['end']]
        cut.export(cut_path, format='wav')

    util_subt.check_coherent_segments(segments)
    util.save_as_json(segments, json_path)
    util_subt.save_segments_as_srt(segments, srt_path, skip_silence=True)
    return json_path


def exec(manager):
    logger.info("segment_split,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    segment_divide_path = manager.get('segment_divide_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_split")
    segment_split_path = segment_split(audio_path, segment_divide_path, output_dir)
    manager['segment_split_path'] = segment_split_path
    logger.info("segment_split,leave: %s", util.json_dumps(manager))
    util.exec_gc()
