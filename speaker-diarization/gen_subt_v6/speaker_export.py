from pydub import AudioSegment
import util
import os

logger = util.get_logger()


def speaker_export(audio_path, speaker_overall_path, output_dir):
    json_path = os.path.join(output_dir, 'speaker_export.json')
    if util.path_exist(json_path):
        return json_path

    speaks = util.read_file_to_obj(speaker_overall_path)
    for i, speak in enumerate(speaks):
        speaks[i].pop('wav_path', None)

    audio = AudioSegment.from_wav(audio_path)
    for i, speak in enumerate(speaks):
        segments = speaks[i]['segments']
        for j, segment in enumerate(segments):
            cut_path = os.path.join(output_dir, 'speaker', speaks[i]['file_name'], f"{segments[j]['file_name']}.wav")
            segments[j]['wav_path'] = cut_path
            util.mkdir(cut_path)
            cut = audio[segments[j]['start']:segments[j]['end']]
            cut.export(cut_path, format='wav')
        speaks[i]['segments'] = segments

    util.save_as_json(speaks, json_path)
    return json_path


def exec(manager):
    logger.info("speaker_export,enter: %s", util.json_dumps(manager))
    speaker_overall_path = manager.get('speaker_overall_path')
    audio_path = manager.get('extract_dereverb_path')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_export")
    speaker_export_path = speaker_export(audio_path, speaker_overall_path, output_dir)
    manager['speaker_export_path'] = speaker_export_path
    logger.info("speaker_export,leave: %s", util.json_dumps(manager))
    util.exec_gc()
