import whisperx
import util
import util_subt
import os
import json
from collections import Counter

logger = util.get_logger()


def gen_subt(audio_path):
    device = util.get_device_type()
    compute_type = util.get_compute_type()

    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    del model
    util.exec_gc()

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    aligned_result["language"] = result["language"]
    del model_a
    util.exec_gc()
    return aligned_result


def gen_and_save_subt(audio_path, output_dir):
    sub = gen_subt(audio_path)
    util_subt.save_subt_as_vtt(audio_path, sub, output_dir)
    util_subt.save_subt_as_json(audio_path, sub, output_dir)
    return sub


def gen_and_join_subt(audio_path, audio_batch_path, audio_split_dir, output_dir):
    split_subt_dir = os.path.join(output_dir, 'split_subt')
    if util.path_exist(split_subt_dir):
        return

    for file in os.listdir(audio_split_dir):
        file_path = os.path.join(audio_split_dir, file)
        if not util.path_isfile(file_path):
            continue
        if 'speech.wav' not in file_path:
            continue
        gen_and_save_subt(file_path, split_subt_dir)

    subtitle = {
        "segments": [],
        "word_segments": [],
        "language": '',
        "languages": [],
    }

    content = util.read_file(audio_batch_path)
    segments = json.loads(content)
    for i, segment in enumerate(segments):
        split_subt_path = os.path.join(split_subt_dir, f'{i:05d}_{segment["vad_type"]}.json')
        if not util.path_exist(split_subt_path):
            continue
        content = util.read_file(split_subt_path)
        subt = json.loads(content)
        subt = util_subt.shift_subt_time(subt, segment['start'])
        subtitle['segments'].append(subt['segments'])
        subtitle['word_segments'].append(subt['word_segments'])
        subtitle['languages'].append(subt['language'])

    if len(subtitle['languages']) > 0:
        counter = Counter(subtitle['languages'])
        most_common = counter.most_common(1)
        subtitle['language'] = most_common[0][0] if most_common else ''

    util_subt.save_subt_as_vtt(audio_path, subtitle, output_dir)
    gen_subt_path = util_subt.save_subt_as_json(audio_path, subtitle, output_dir)
    return gen_subt_path


def gen_subt_by_manager(manager):
    logger.info("gen_subt,enter,manager: %s", json.dumps(manager))
    merge_audio_channel_path = manager.get('merge_audio_channel_path')
    audio_batch_path = manager.get('audio_batch_path')
    audio_split_dir = manager.get('audio_split_dir')
    output_dir = os.path.join(manager.get('output_dir'), "gen_subt")
    gen_subt_path =gen_and_join_subt(merge_audio_channel_path, audio_batch_path, audio_split_dir, output_dir)
    manager['gen_subt_path'] = gen_subt_path
    logger.info("gen_subt,leave,manager: %s", json.dumps(manager))
