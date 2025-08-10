import speaker_detect_pyannote
import os
import util
import json

logger = util.get_logger()


def speaker_detect(speak_path, output_dir, auth_token, window=10, step=5):
    json_path = os.path.join(output_dir, 'speaker_detect.json')
    if util.path_exist(json_path):
        return json_path

    content = util.read_file(speak_path)
    speaks = json.loads(content)
    if len(speaks) < window:
        window = len(speaks)

    groups = []
    left = 0
    right = left + window
    while right <= len(speaks):
        if right + step > len(speaks):
            right = len(speaks)
        result = speaker_detect_pyannote.speaker_detect(speaks[left:right], auth_token)
        groups.extend(result)
        left += step
        right += step

    util.save_as_json(groups, json_path)
    return json_path
