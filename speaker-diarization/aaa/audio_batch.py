import util
import json
import util_sub
import os
import math

logger = util.get_logger()


def audio_batch(audio_activity_path, output_dir, min_silene_duration_ms=500, min_speech_duration_ms=1000 * 5):
    audio_batch_path = os.path.join(output_dir, 'audio_batch.json')
    if util.path_exist(audio_batch_path):
        return audio_batch_path

    content = util.read_file(audio_activity_path)
    segments = json.loads(content)
    batches = []
    for i, segment in enumerate(segments):
        if len(batches) == 0:
            batches.append({"start": 0, "end": 0})
            continue
        if segment['vad_type'] != 'silene':
            continue
        if segment['end'] - segment['start'] < min_silene_duration_ms:
            continue
        if segment['end'] - batches[len(batches) - 1]['start'] < min_speech_duration_ms:
            continue
        mean = math.floor((segments[i]['end'] + segments[i]['start']) / 2.0)
        batches[len(batches) - 1]['end'] = mean
        batches.append({"start": mean, "end": 0})
    batches.pop()
    batches[len(batches) - 1]['end'] = segments[len(segments) - 1]['end']
    util_sub.check_segments(batches)
    segments = batches

    util.save_file(json.dumps(segments), audio_batch_path)
    util_sub.save_segments_as_srt(segments, os.path.join(output_dir, 'audio_batch.srt'), skip_silene=True)
    return audio_batch_path


def audio_batch_by_manager(manager):
    logger.info("audio_batch,enter,manager: %s", json.dumps(manager))
    audio_activity_path = manager.get('audio_activity_path')
    output_dir = os.path.join(manager.get('output_dir'), "audio_batch")
    audio_batch_path = audio_batch(audio_activity_path, output_dir)
    manager['audio_batch_path'] = audio_batch_path
    logger.info("audio_batch,leave,manager: %s", json.dumps(manager))
