import util
import os

logger = util.get_logger()


class Extract:
    def __init__(self, handler):
        self.handler = handler

    def extract(self, audio_path, output_dir):
        handler_dir = os.path.join(output_dir, self.handler.get_name())
        master_path, slave_path = self.handler.extract(audio_path, handler_dir)
        self.master_path = master_path
        self.slave_path = slave_path
        self.master_copy_path = os.path.join(output_dir, self.handler.get_master_name())
        self.slave_copy_path = os.path.join(output_dir, self.handler.get_slave_name())

    def copy_master(self):
        util.copy_file(self.master_path, self.master_copy_path)

    def copy_slave(self):
        util.copy_file(self.slave_path, self.slave_copy_path)


def extract_stem(audio_path, handlers, output_dir):
    path_map = {}
    extracts = []
    for i, handler in enumerate(handlers):
        extracts.append(Extract(handler))
    for i, extract in enumerate(extracts):
        extract.extract(audio_path, output_dir)
        extract.copy_slave()
        slave_copy_path = extract.slave_copy_path
        file_name = util.get_file_name(slave_copy_path)
        path_map[f'{file_name}_path'] = slave_copy_path
        audio_path = extract.master_path
    if len(extracts) > 0:
        extracts[-1].copy_master()
        master_copy_path = extracts[-1].master_copy_path
        file_name = util.get_file_name(master_copy_path)
        path_map[f'{file_name}_path'] = master_copy_path
        path_map['audio_path'] = master_copy_path
        path_map['split_audio_path'] = master_copy_path
    return path_map


def exec(manager, handlers):
    logger.info("extract_stem,enter: %s", util.json_dumps(manager))
    audio_path = manager.get('audio_path')
    output_dir = os.path.join(manager.get('output_dir'), "extract_stem")
    path_map = extract_stem(audio_path, handlers, output_dir)
    manager['extract_stem_path_map'] = path_map
    manager['audio_path'] = path_map['audio_path']
    manager['split_audio_path'] = path_map['split_audio_path']
    logger.info("extract_stem,leave: %s", util.json_dumps(manager))
    util.exec_gc()
