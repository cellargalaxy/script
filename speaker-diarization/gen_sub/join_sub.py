import json
from whisperx.utils import get_writer
import util
import os
import sub_util
from collections import Counter
from pydub import AudioSegment


def join_sub(audio_dir, sub_dir):
    subtitle = {
        "segments": [],
        "word_segments": [],
        "language": '',
        "languages": [],
    }
    start = 0
    for file in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, file)
        if not util.path_isfile(audio_path):
            continue
        if '.wav' not in audio_path:
            continue
        audio = AudioSegment.from_wav(audio_path)
        audio_len = len(audio)
        del audio
        util.exec_gc()

        json_path = os.path.join(sub_dir, util.get_file_name(audio_path) + '.json')
        if not util.path_exist(json_path):
            start = start + audio_len
            continue

        content = util.read_file(json_path)
        sub = json.loads(content)

        segments = sub.get('segments', [])
        for i, segment in enumerate(segments):
            segments[i]['start'] = segments[i]['start'] + (start / 1000.0)
            segments[i]['end'] = segments[i]['end'] + (start / 1000.0)
            words = segments[i].get('words', [])
            for j, word in enumerate(words):
                words[j]['start'] = words[j]['start'] + (start / 1000.0)
                words[j]['end'] = words[j]['end'] + (start / 1000.0)
            segments[i]['words'] = words
        sub['segments'] = segments

        word_segments = sub.get('word_segments', [])
        for i, word_segment in enumerate(word_segments):
            word_segments[i]['start'] = word_segments[i]['start'] + (start / 1000.0)
            word_segments[i]['end'] = word_segments[i]['end'] + (start / 1000.0)
        sub['word_segments'] = word_segments

        subtitle['segments'].extend(sub['segments'])
        subtitle['word_segments'].extend(sub['word_segments'])
        language = sub.get('language', '')
        if language:
            subtitle['languages'].append(language)

        start = start + audio_len

    if len(subtitle['languages']) > 0:
        counter = Counter(subtitle['languages'])
        most_common = counter.most_common(1)
        subtitle['language'] = most_common[0][0] if most_common else ''
    return subtitle


def join_sub_and_save(audio_path, audio_dir, sub_dir, save_dir):
    subtitle = join_sub(audio_dir, sub_dir)
    sub_util.save_sub_as_vtt(audio_path, subtitle, save_dir)
    sub_util.save_sub_as_json(audio_path, subtitle, save_dir)


def join_sub_and_save_by_manager(manager):
    audio_path = manager.get('audio_path')
    audio_dir = manager.get('audio_split_batch_dir')
    sub_dir = manager.get('transcribe_sub_dir')
    save_dir = os.path.join(manager.get('output_dir'), "join_sub")
    join_sub_and_save(audio_path, audio_dir, sub_dir, save_dir)
    manager['join_sub_dir'] = save_dir
