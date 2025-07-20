import util
import json
import os
from pydub import AudioSegment
import util_subt

logger = util.get_logger()


def segment_split(audio_path, segment_divide_path, output_dir):
    json_path = os.path.join(output_dir, 'segment_split.json')
    srt_path = os.path.join(output_dir, 'segment_split.srt')
    split_dir = os.path.join(output_dir, 'split')
    if util.path_exist(json_path):
        return json_path

    content = util.read_file(segment_divide_path)
    segments = json.loads(content)

    for i, segment in enumerate(segments):
        segments[i]['file_name'] = f"{i:05d}_{segment['vad_type']}"
        segments[i]['wav_name'] = f"{segments[i]['file_name']}.wav"

    util.mkdir(output_dir)
    audio = AudioSegment.from_wav(audio_path)
    for i, segment in enumerate(segments):
        cut_path = os.path.join(output_dir, segment['wav_name'])
        cut = audio[segment['start']:segment['end']]
        cut.export(cut_path, format='wav')

    util_subt.check_coherent_segments(segments)
    util.save_as_json(segments, json_path)
    util_subt.save_segments_as_srt(segments, srt_path, skip_silene=True)
    return json_path, split_dir


def exec(manager):
    logger.info("segment_split,enter: %s", json.dumps(manager))
    audio_path = manager.get('merge_channel_path')
    segment_divide_path = manager.get('segment_divide_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_split")
    json_path, split_dir = segment_split(audio_path, segment_divide_path, output_dir)
    manager['segment_split_json_path'] = json_path
    manager['segment_split_split_dir'] = split_dir
    logger.info("segment_split,leave: %s", json.dumps(manager))
