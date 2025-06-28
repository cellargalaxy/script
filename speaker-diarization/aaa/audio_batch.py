import util
import json
import util_sub
import os

logger = util.get_logger()


def audio_batch(audio_activity_path, output_dir, min_silene_duration_ms=500, min_speech_duration_ms=1000 * 10):
    audio_batch_path = os.path.join(output_dir, 'audio_batch.json')
    if util.path_exist(audio_batch_path):
        return audio_batch_path

    content = util.read_file(audio_activity_path)
    segments = json.loads(content)
    segments = util_sub.gradual_segments(segments, gradual_duration_ms=min_silene_duration_ms)
    util.save_file(json.dumps(segments), os.path.join(output_dir, 'audio_batch_gradual.json'))
    util_sub.save_segments_as_srt(segments, os.path.join(output_dir, 'audio_batch_gradual.srt'), skip_silene=True)

    batches = []
    for i, segment in enumerate(segments):
        if len(batches) == 0:
            batches.append({"start": 0, "end": 0})
            continue
        if segment['end'] - batches[len(batches) - 1]['start'] < min_speech_duration_ms:
            continue
        if segment['vad_type'] != 'silene':
            continue
        if segment['vad_type'] == 'silene':
            batches[len(batches) - 1]['end'] = segments[i]['start']
            batches.append(segments[i])
            batches.append({"start": segments[i]['end'], "end": 0})
        else:
            batches[len(batches) - 1]['end'] = segments[i]['end']
            batches.append({"start": segments[i]['end'], "end": 0})
    batches.pop()
    batches[len(batches) - 1]['end'] = segments[len(segments) - 1]['end']
    batches[len(batches) - 1]['vad_type'] = 'silene'
    for i, segment in enumerate(segments):
        if segment['start'] < batches[len(batches) - 1]['start']:
            continue
        if segment['vad_type'] == 'speech':
            batches[len(batches) - 1]['vad_type'] = 'speech'
            break
    segments = []
    for i, segment in enumerate(batches):
        if segment['start'] == segment['end']:
            continue
        segments.append(segment)
    util_sub.check_segments(segments)
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
