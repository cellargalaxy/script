import util
import os

logger = util.get_logger()


def exec(manager):
    util.print_sys_info()
    logger.info("init,enter: %s", util.json_dumps(manager))
    video_path = manager.get('video_path')
    file_name = util.get_file_name(video_path)
    output_dir = os.path.join('output', file_name)
    manager['output_dir'] = output_dir
    logger.info("init,leave: %s", util.json_dumps(manager))
