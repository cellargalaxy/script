import whisper_timestamped as whisper
from whisperx.utils import get_writer
import util
import os
import gc
import torch


def whisper_timestamped(audio_path, device, model_name="large-v3"):
    audio = whisper.load_audio(audio_path)
    model = whisper.load_model(model_name, device=device)
    result = whisper.transcribe(model, audio)
    del audio
    del model
    torch.cuda.empty_cache()
    gc.collect()
    whisper_result = {
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
            whisper_result['word_segments'].append(obj)
        obj = {
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'],
            "words": words,
            "speaker": '',
        }
        whisper_result['segments'].append(obj)
    file_dir = util.get_file_dir(audio_path)
    vtt_writer = get_writer("vtt", file_dir)
    vtt_writer(
        whisper_result,
        audio_path,
        {"max_line_width": None, "max_line_count": None, "highlight_words": True},
    )


def whisper_timestamped_by_manager(manager):
    device = manager.get('device')
    audio_dir = manager.get('split_video_dir')
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        if not os.path.isfile(file_path):
            continue
        whisper_timestamped(file_path, device)
