import visitor
import numpy as np
import torch


class SileroVadStreamFilterMuteVisitor(visitor.Visitor):
    model = None
    chunk_size = 0
    window_len = 0
    mute = True
    data_window = None

    def __init__(self):
        self.model, _ = torch.hub.load(repo_or_dir='model/silero-vad/master', model='silero_vad', source='local')
        self.chunk_size = visitor.get_chunk_byte_size()
        self.window_len = visitor.get_second_byte_size(0.2)
        self.confidences = []

    def start(self, data):
        if not self.mute:
            self.start_next(data)

    def exec(self, data):
        if not data:
            return
        if self.data_window is None:
            self.data_window = data
        else:
            self.data_window += data

        # 有说话->有说话：不用处理
        # 有说话->没说话：mute=True即可
        # 没说话->有说话：mute=False，将前几个data补上，最后exec_next_mute里写入
        # 没说话->没说话：不用处理

        if len(self.data_window) >= self.window_len:
            speaking = self.is_speaks(self.data_window)
            if not self.mute and not speaking:
                print(f"\n没说话")
                self.mute = True
            if self.mute and speaking:
                print(f"\n有说话")
                self.mute = False
                self.exec_next_mute(self.data_window[:-len(data)])
            self.data_window = None
        self.exec_next_mute(data)

    def is_speaks(self, data_window):
        datas = [data_window[i:i + self.chunk_size] for i in range(0, len(data_window), self.chunk_size)]
        step = visitor.get_second_chunk_count(0.5)
        if step <= 0:
            step = 1
        indexs = []
        for i in range(0, len(datas), step):
            indexs.append(i)
        indexs.append(len(datas) - 1)
        if len(indexs) <= 2:
            indexs = [len(datas) - 1]
        indexs = list(set(indexs))
        data_list = []
        for index in indexs:
            data_list.append(datas[index])

        confidences = []
        for data in data_list:
            confidence = self.confidence(data)
            confidences.append(confidence)
        print(f"\rconfidences: {confidences}", end="")
        for confidence in confidences:
            if confidence >= 0.2:
                return True
        return False

    def confidence(self, data):
        data_int16 = np.frombuffer(data, np.int16)
        data_float32 = self.int2float(data_int16)
        confidence = self.model(torch.from_numpy(data_float32), visitor.SAMPLE_RATE).item()
        return confidence

    def exec_next_mute(self, data):
        if not self.mute:
            self.exec_next(data)

    def stop(self, data):
        if not self.mute:
            self.stop_next(data)

    def int2float(self, data):
        abs_max = np.abs(data).max()
        data = data.astype('float32')
        if abs_max > 0:
            data *= 1 / 32768
        data = data.squeeze()
        return data
