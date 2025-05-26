import json
from whisperx.utils import get_writer
import util
import os

logger = util.get_logger()


def check_segments(segments):
    for i, segment in enumerate(segments):
        start = segments[i].get('start', -1)
        if not isinstance(start, int):
            logger.error("检查segments，start类型非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，start类型非法")
        if start < 0:
            logger.error("检查segments，start非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，start非法")
        end = segments[i].get('end', -1)
        if not isinstance(end, int):
            logger.error("检查segments，end类型非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，end类型非法")
        if end < 0:
            logger.error("检查segments，end非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，end非法")
        if end <= start:
            logger.error("检查segments，start与end非法: %s, %s", i, json.dumps(segments))
            raise ValueError("检查segments，start与end非法")
        if i != 0:
            pre_end = segments[i - 1]['end']
            if start <= pre_end:
                logger.error("检查segments，pre_end与start非法: %s, %s", i, json.dumps(segments))
                raise ValueError("检查segments，pre_end与start非法")


def save_sub(file_path, sub_result):
    file_dir = util.get_file_dir(file_path)
    vtt_writer = get_writer("vtt", file_dir)
    vtt_writer(
        sub_result,
        file_path,
        {"max_line_width": None, "max_line_count": None, "highlight_words": True},
    )
    json_path = os.path.join(file_dir, util.get_file_name(file_path) + '.json')
    util.save_file(json.dumps(sub_result), json_path)
