import util
import json
import speaker_detect_pyannote

logger = util.get_logger()


def speaker_detect(input_dir, output_dir, auth_token):
    json_path = os.path.join(output_dir, 'speaker_detect.json')
    if util.path_exist(json_path):
        return json_path

    speaks = []
    files = util.listdir(input_dir)
    for file in files:
        speak = {
            "file_name": util.get_file_name(file),
            "wav_path": os.path.join(input_dir, file)
        }
        speaks.append(speak)

    groups = speaker_detect_pyannote.speaker_detect(speaks, auth_token=auth_token)
    groups = [sorted(group) for group in groups]
    groups = sorted(groups, key=lambda x: len(x), reverse=True)

    for i, group in enumerate(groups):
        for j, file_name in enumerate(group):
            input_path = os.path.join(input_dir, f"{file_name}.wav")
            output_path = os.path.join(output_dir, f"{i:02}_{len(group):03}", f"{file_name}.wav")
            util.copy_file(input_path, output_path)

    util.save_as_json(groups, json_path)
    return json_path


def exec(manager):
    logger.info("speaker_detect,enter: %s", json.dumps(manager))
    input_dir = manager.get('input_dir')
    auth_token = manager.get('auth_token')
    output_dir = os.path.join(manager.get('output_dir'), "speaker_detect")
    speaker_detect_path = speaker_detect(input_dir, output_dir, auth_token)
    manager['speaker_detect_path'] = speaker_detect_path
    logger.info("speaker_detect,leave: %s", json.dumps(manager))
    util.exec_gc()


if __name__ == '__main__':
    import os

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['http_proxy'] = 'http://192.168.123.7:10808'
    os.environ['https_proxy'] = 'http://192.168.123.7:10808'
    os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

    input_dir = "../material/shigeju/raw_wav"
    manager = {
        "input_dir": input_dir,
        "output_dir": util.get_ancestor_dir(input_dir),
        "auth_token": "",
    }
    exec(manager)
