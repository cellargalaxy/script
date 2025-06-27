import util
import json
import util_sub
import os

logger = util.get_logger()


def audio_batch(audio_activity_path, output_dir, min_silene_duration_ms=500, min_speech_duration_ms=1000 * 5):
    content = util.read_file(audio_activity_path)
    segments = json.loads(content)
    segments = util_sub.gradual_segments(segments, gradual_duration_ms=min_silene_duration_ms)
    # batches = []
    # for i, segment in enumerate(segments):
    #     if len(batches) == 0:
    #         batches.append({"start": 0, "end": 0})
    #         continue
    #     start = batches[len(batches) - 1]['start']
    #     end = segment['end']
    #     duration = end - start
    #     if min_speech_duration_ms <= duration and segment['vad_type'] == 'silene':
    #         batches[len(batches) - 1]['end'] = end
    #         batches.append({"start": end, "end": 0})
    # batches.pop()
    # batches[len(batches) - 1]['end'] = segments[len(segments) - 1]['end']
    # util_sub.check_segments(batches)
    # segments = batches

    audio_batch_path = os.path.join(output_dir, 'audio_batch.json')
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
