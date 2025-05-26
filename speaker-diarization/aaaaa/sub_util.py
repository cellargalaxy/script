import json
from whisperx.utils import get_writer
import util
import os


def save_sub(file_path,sub_result):
    file_dir = util.get_file_dir(file_path)
    vtt_writer = get_writer("vtt", file_dir)
    vtt_writer(
        sub_result,
        file_path,
        {"max_line_width": None, "max_line_count": None, "highlight_words": True},
    )
    json_path = os.path.join(file_dir, util.get_file_name(file_path) + '.json')
    util.save_file(json.dumps(sub_result), json_path)
