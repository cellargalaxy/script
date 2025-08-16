import speaker_detect_pyannote
import os
import util

logger = util.get_logger()


def speaker_detect(speak_path, output_dir, auth_token, window=16, step=8):
    json_path = os.path.join(output_dir, 'speaker_detect.json')
    if util.path_exist(json_path):
        return json_path

    speaks = util.read_file_to_obj(speak_path)
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

    groups = [sorted(inner) for inner in groups]
    groups = sorted(groups, key=lambda x: len(x), reverse=True)

    util.save_as_json(groups, json_path)
    return json_path
