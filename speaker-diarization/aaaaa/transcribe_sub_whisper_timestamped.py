import whisper_timestamped as whisper
import util
import os
import sub_util


def transcribe_sub(audio_path):
    model = whisper.load_model("large-v3", device=util.get_device_type())
    audio = whisper.load_audio(audio_path)
    result = whisper.transcribe(model, audio)
    del model
    del audio
    util.exec_gc()
    sub = {
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
            }
            words.append(obj)
            sub['word_segments'].append(obj)
        obj = {
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'],
            "words": words,
        }
        sub['segments'].append(obj)
    return sub


def transcribe_and_save_sub(audio_path, output_dir):
    sub = transcribe_sub(audio_path)
    sub_util.save_sub_as_vtt(audio_path, sub, output_dir)
    sub_util.save_sub_as_json(audio_path, sub, output_dir)


def transcribe_and_save_sub_by_manager(manager):
    audio_dir = manager.get('split_video_dir')
    output_dir = os.path.join(manager.get('output_dir'), "transcribe_sub")
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        if not util.path_isfile(file_path):
            continue
        if 'speech.wav' not in file_path:
            continue
        transcribe_and_save_sub(file_path, output_dir)
    manager['transcribe_sub_dir'] = output_dir
