import copy
import os
from speechbrain.inference.speaker import SpeakerRecognition
import random
import util
import json

logger = util.get_logger()


def is_audio_same(a_path, b_path):
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir="model/spkrec-ecapa-voxceleb")
    score, prediction = verification.verify_files(a_path, b_path)
    del verification
    util.exec_gc()
    return prediction


def sample_inspection(audio_path, audio_paths):
    simple_paths = copy.deepcopy(audio_paths)
    for i, simple_path in enumerate(simple_paths):
        if simple_path == audio_path:
            simple_paths.pop(i)
            break
    simple_num = min(10, len(simple_paths))
    simples = []
    for i in range(simple_num):
        simple = simple_paths.pop(random.randrange(len(simple_paths)))
        simples.append(simple)
    return simples


def audio_purification(audio_dir, output_dir):
    audio_paths = []
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        if not util.path_isfile(file_path):
            audio_purification(file_path, output_dir)
            continue
        audio_paths.append(file_path)
    for i, audio_path in enumerate(audio_paths):
        simple_paths = sample_inspection(audio_path, audio_paths)
        diff_cnt = 0.0
        for i, simple_path in enumerate(simple_paths):
            same = is_audio_same(audio_path, simple_path)
            if not same:
                diff_cnt += 1
        same = diff_cnt / len(simple_paths) < 0.5
        if same:
            speaker = util.get_parent_dir(audio_path)
            output_path = os.path.join(output_dir, speaker, util.get_file_basename(audio_path))
            util.copy_file(audio_path, output_path)


def audio_purification_by_manager(manager):
    logger.info("audio_purification,enter,manager: %s", json.dumps(manager))
    audio_dir = manager.get('audio_class_dir')
    output_dir = os.path.join(manager.get('output_dir'), "audio_purification")
    util.delete_path(output_dir)
    audio_purification(audio_dir, output_dir)
    manager['audio_purification_dir'] = output_dir
    logger.info("audio_purification,leave,manager: %s", json.dumps(manager))
