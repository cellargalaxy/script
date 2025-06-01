import whisperx
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
    aligned_result["language"] = result["language"]
    del model_a
    util.exec_gc()
    # return aligned_result

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=auth_token, device=device)
    diarize_segments = diarize_model(audio)
    diarize_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
    diarize_result["language"] = result["language"]

    return diarize_result


def transcribe_and_save_sub(audio_path, output_dir, auth_token=''):
    sub = transcribe_sub(audio_path, auth_token=auth_token)
    sub_util.save_sub_as_vtt(audio_path, sub, output_dir)
    sub_util.save_sub_as_json(audio_path, sub, output_dir)


def transcribe_and_save_sub_by_manager(manager):
    audio_dir = manager.get('audio_split_batch_dir')
    output_dir = os.path.join(manager.get('output_dir'), "transcribe_sub")
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        if not util.path_isfile(file_path):
            continue
        if 'speech.wav' not in file_path:
            continue
        transcribe_and_save_sub(file_path, output_dir)
    manager['transcribe_sub_dir'] = output_dir
