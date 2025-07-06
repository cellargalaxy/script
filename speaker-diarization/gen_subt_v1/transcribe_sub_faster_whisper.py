from faster_whisper import WhisperModel
import util
import os
import sub_util


def transcribe_sub(audio_path):
    device = util.get_device_type()
    compute_type = util.get_compute_type()

    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, beam_size=8)
    del model
    util.exec_gc()
    sub = {
        "segments": [],
        "language": info.language,
    }
    for segment in segments:
        obj = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        sub['segments'].append(obj)
    return sub


def transcribe_and_save_sub(audio_path, output_dir):
    sub = transcribe_sub(audio_path)
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
