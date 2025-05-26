import whisper_timestamped as whisper
import util
import os
import gc
import torch
import sub_util


def transcribe_sub(audio_path):
    model = whisper.load_model("large-v3", device=util.get_device_type())
    audio = whisper.load_audio(audio_path)
    result = whisper.transcribe(model, audio)
    del model
    del audio
    util.exec_gc()
    sub_result = {
        "segments": [],
        "word_segments": [],
        "language": result["language"],
    }
    segments = result['segments']
    for segment in segments:
        words = []
        for word in segment['words']:
            obj = {
                "word": word['text'],
                "start": word['start'],
                "end": word['end'],
                "score": word['confidence'],
                "speaker": '',
            }
            words.append(obj)
            sub_result['word_segments'].append(obj)
        obj = {
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'],
            "words": words,
            "speaker": '',
        }
        sub_result['segments'].append(obj)
    return sub_result


def transcribe_and_save_sub(audio_path):
    sub_result = transcribe_sub(audio_path)
    sub_util.save_sub(sub_result, audio_path)


def transcribe_and_save_sub_by_manager(manager):
    audio_dir = manager.get('split_video_dir')
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        if not util.path_isfile(file_path):
            continue
        if '_speech.' not in file_path:
            continue
        transcribe_and_save_sub(file_path)
