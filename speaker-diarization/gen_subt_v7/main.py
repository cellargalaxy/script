import os

import util
import split_audio
import init
import extract_audio
import extract_stem
import extract_stem_uvr
import extract_loudness
import extract_simple
import part_detect
import part_divide
import segment_detect
import segment_divide
import speaker_detect

logger = util.get_logger()


def exec(video_path, exec_conf):
    manager = {"video_path": video_path}
    for i, conf in enumerate(exec_conf):
        if not conf.get('enable', False):
            continue
        name = conf.get('name', None)
        if name == 'init':
            init.exec(manager)
        if name == 'extract_audio':
            extract_audio.exec(manager)
        if name == 'extract_stem':
            vocal_model = conf.get('vocal_model', None)
            main_vocal_model = conf.get('main_vocal_model', None)
            de_reverb_model = conf.get('de_reverb_model', None)
            extract_stem.exec(manager, [
                extract_stem_uvr.VocalHandler(vocal_model),
                extract_stem_uvr.MainVocalHandler(main_vocal_model),  # 5_HP-Karaoke-UVR.pth
                extract_stem_uvr.DeReverbHandler(de_reverb_model),  # MDX23C-De-Reverb-aufr33-jarredou.ckpt
            ])
        if name == 'extract_loudness':
            extract_loudness.exec(manager)
        if name == 'extract_simple':
            extract_simple.exec(manager)
        if name == 'part_detect':
            part_detect.exec(manager)
        if name == 'part_divide':
            part_divide.exec(manager)
        if name == 'segment_detect':
            segment_detect.exec(manager)
        if name == 'segment_divide':
            segment_divide.exec(manager)
        if name == 'speaker_detect':
            speaker_detect.exec(manager)
        if name == 'split_audio':
            path_key = conf.get('path_key', None)
            split_audio.exec(manager, path_key)


if __name__ == '__main__':
    json_path = os.path.join(util.get_ancestor_dir(util.get_script_path()), 'main.json')
    conf = util.read_file_to_obj(json_path)

    env = conf.get('video', None) or {}
    for key, value in env.items():
        os.environ[key] = value

    video_paths = conf.get('video', None) or []
    exec_conf = conf.get('exec', None) or []
    for i, video_path in enumerate(video_paths):
        try:
            exec(video_path, exec_conf)
        except Exception as e:
            logger.error("异常", exc_info=True)
            util.input_timeout("异常，回车继续: ", 60)
