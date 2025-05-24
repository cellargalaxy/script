import os
import util
import ffprobe_util
import ffmpeg_util


def split_video(input_file, output_dir, window, overlap):
    ext = util.get_file_ext(input_file)
    duration = ffprobe_util.get_video_duration(input_file)
    step = window - overlap
    index = 0
    start = 0
    while start + window <= duration:
        end = start + window
        if duration - end < window:
            end = duration
        output_path = os.path.join(output_dir, f'{index:03d}.{ext}')
        ffmpeg_util.cut_video(input_file, start, end, output_path)
        start = start + step


def split_video_by_manager(manager):
    input_file = manager.get('demucs_video_path')
    output_dir = os.path.join(manager.get('output_dir'), "split_video")
    window = manager.get('split_video_window')
    overlap = manager.get('split_video_overlap')
    split_video(input_file, output_dir, window, overlap)
