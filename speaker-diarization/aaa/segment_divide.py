import util
import json
import util_subt
import os
from pydub import AudioSegment

logger = util.get_logger()


def segment_divide(audio_path, subt_gen_path, output_dir,
                   min_silene_duration_ms=500,
                   min_speech_duration_ms=1000,
                   silene_duration_ms=1000):
    json_path = os.path.join(output_dir, 'segment_divide.json')
    srt_path = os.path.join(output_dir, 'segment_divide.srt')
    if util.path_exist(json_path):
        return json_path

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    content = util.read_file(subt_gen_path)
    subt = json.loads(content)
    segments = util_subt.subt2segments(subt)
    if len(segments) > 0 and last_end < segments[-1]['end']:
        segments[-1]['end'] = last_end
    gradual = []
    for i, segment in enumerate(segments):
        if segment['end'] - segment['start'] < min_speech_duration_ms:
            continue
        gradual.append(segment)
    for i, segment in enumerate(gradual):
        gradual[i]['vad_type'] = 'speech'
    segments = []
    for i, segment in enumerate(gradual):
        pre_end = 0
        if i > 0:
            pre_end = gradual[i - 1]['end']
        if pre_end < gradual[i]['start']:
            segments.append({"start": pre_end, "end": gradual[i]['start'], "vad_type": 'silene'})
        segments.append(gradual[i])
    if len(segments) > 0 and segments[-1]['end'] < last_end:
        segments.append({"start": segments[-1]['end'], "end": last_end, "vad_type": 'silene'})
    gradual = segments
    util_subt.check_coherent_segments(gradual)
    gradual = util_subt.gradual_segments(gradual, gradual_duration_ms=min_silene_duration_ms, audio_data=audio)
    util.save_file(json.dumps(gradual), json_path)
    util_subt.save_segments_as_srt(gradual, srt_path, skip_silene=True)

    # arrange = []
    # for i, segment in enumerate(gradual):
    #     if segment['vad_type'] == 'silene':
    #         continue
    #     pre_end = 0
    #     if len(arrange) > 0:
    #         pre_end = arrange[-1]['end']
    #     arrange.append({
    #         "start": pre_end,
    #         "end": pre_end + silene_duration_ms,
    #         "vad_type": 'silene',
    #     })
    #     duration = segment['end'] - segment['start']
    #     arrange.append({
    #         "start": pre_end + silene_duration_ms,
    #         "end": pre_end + silene_duration_ms + duration,
    #         "vad_type": 'speech',
    #         "text": segment['text'],
    #     })
    # if len(arrange) > 0:
    #     pre_end = arrange[-1]['end']
    #     arrange.append({
    #         "start": pre_end,
    #         "end": pre_end + silene_duration_ms,
    #         "vad_type": 'silene',
    #     })
    # util_subt.check_coherent_segments(arrange)
    # util.save_file(json.dumps(arrange), os.path.join(output_dir, 'arrange.json'))
    # util_subt.save_segments_as_srt(arrange, os.path.join(output_dir, 'arrange.srt'), skip_silene=True)
    #
    # blank_data = AudioSegment.silent(duration=silene_duration_ms)
    # arrange_data = AudioSegment.silent(duration=silene_duration_ms)
    # for i, segment in enumerate(gradual):
    #     if segment['vad_type'] == 'silene':
    #         continue
    #     cut = audio[segment['start']:segment['end']]
    #     arrange_data = arrange_data + cut + blank_data
    # arrange_path = os.path.join(output_dir, 'arrange.wav')
    # util.mkdir(arrange_path)
    # arrange_data.export(arrange_path, format="wav")

    return json_path


def segment_divide_by_manager(manager):
    logger.info("segment_divide,enter,manager: %s", json.dumps(manager))
    audio_path = manager.get('merge_audio_channel_path')
    subt_gen_path = manager.get('subt_gen_path')
    output_dir = os.path.join(manager.get('output_dir'), "segment_divide")
    segment_divide_path = segment_divide(audio_path, subt_gen_path, output_dir)
    manager['segment_divide_path'] = segment_divide_path
    logger.info("segment_divide,leave,manager: %s", json.dumps(manager))
