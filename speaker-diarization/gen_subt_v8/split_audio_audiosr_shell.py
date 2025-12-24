import util
import os

logger = util.get_logger()


def exec(manager, path_key, min_duration=None):
    py_path = os.path.join(util.get_script_path(), "split_audio_audiosr.py")
    manager_json = util.json_dumps(manager)
    cmd = [
        "conda", "run", "-n", "audiosr",
        "python", py_path,
        f"manager={manager_json}",
        f"path_key={path_key}",
        f"min_duration={min_duration}",
    ]
    logger.info("split_audio_audiosr,cmd: %s", util.json_dumps(cmd))
    return_code = util.popen_cmd(cmd)
    if return_code != 0:
        logger.error("split_audio_audiosr，异常")
        raise ValueError("split_audio_audiosr，异常")
