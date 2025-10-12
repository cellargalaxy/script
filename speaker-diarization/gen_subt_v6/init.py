import util
import os

logger = util.get_logger()


def init_shell():
    cmd = [
        'sh', 'init.sh',
    ]
    logger.info("初始化,cmd: %s", util.json_dumps(cmd))
    stdout, return_code = util.run_cmd(cmd)
    if return_code != 0:
        logger.error("初始化，异常")
        raise ValueError("初始化，异常")


def exec(manager):
    logger.info("init,enter: %s", util.json_dumps(manager))
    init_shell()
    video_path = manager.get('video_path')
    file_name = util.get_file_name(video_path)
    output_dir = os.path.join('output', file_name)
    manager['output_dir'] = output_dir
    logger.info("init,leave: %s", util.json_dumps(manager))
    util.exec_gc()
