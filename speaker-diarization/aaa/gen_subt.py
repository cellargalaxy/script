import whisperx
import util
import util_subt
import os
import json

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


def gen_and_save_subt_dir(audio_dir, output_dir):
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        if not util.path_isfile(file_path):
            continue
        if 'speech.wav' not in file_path:
            continue
        gen_and_save_subt(file_path, output_dir)


def gen_subt_by_manager(manager):
    logger.info("gen_subt,enter,manager: %s", json.dumps(manager))
    audio_split_dir = manager.get('audio_split_dir')
    output_dir = os.path.join(manager.get('output_dir'), "gen_subt")
    gen_and_save_subt_dir(audio_split_dir, output_dir)
    manager['gen_subt_dir'] = output_dir
    logger.info("gen_subt,leave,manager: %s", json.dumps(manager))
