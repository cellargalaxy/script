import whisperx
import gc
import torch
import util
import sub_util
import os


def transcribe_sub(audio_path, auth_token=''):
    device = util.get_device_type()
    compute_type = util.get_compute_type()

    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    del model
    util.exec_gc()

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    del model_a
    util.exec_gc()

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=auth_token, device=device)
    diarize_segments = diarize_model(audio)
    diarize_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
    diarize_result["language"] = result["language"]

    return diarize_result


def transcribe_and_save_sub(audio_path, auth_token=''):
    sub_result = transcribe_sub(audio_path, auth_token=auth_token)
    sub_util.save_sub(sub_result, audio_path)


def transcribe_and_save_sub_by_manager(manager):
    audio_dir = manager.get('split_video_dir')
    auth_token = manager.get('auth_token')
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        if not util.path_isfile(file_path):
            continue
        if '_speech.' not in file_path:
            continue
        transcribe_and_save_sub(file_path, auth_token=auth_token)
