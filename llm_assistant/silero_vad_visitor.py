import visitor
import numpy as np
import torch
import pyaudio
import matplotlib.pyplot as plt


class SileroVadStreamFilterMuteVisitor(visitor.Visitor):
    model = None
    data_window = None
    is_plot = False
    confidences = None

    def __init__(self, is_plot=False):
        self.model, utils = torch.hub.load(repo_or_dir='./model/silero-vad/master', model='silero_vad', source='local')
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        self.is_plot = is_plot
        self.confidences = []

    def start(self, data):
        self.start_next(data)

    def exec(self, data):
        if not data:
            return
        if self.data_window is None:
            self.data_window = data
        else:
            self.data_window += data

        match visitor.FORMAT:
            case pyaudio.paInt8:
                chunk_size = visitor.CHUNK_SIZE
            case pyaudio.paInt16:
                chunk_size = visitor.CHUNK_SIZE * 2
            case pyaudio.paInt24:
                chunk_size = visitor.CHUNK_SIZE * 3
            case pyaudio.paInt32:
                chunk_size = visitor.CHUNK_SIZE * 4
            case _:
                raise TypeError("未支持的音频流格式")

        datas = [self.data_window[i:i + chunk_size] for i in range(0, len(self.data_window), chunk_size)]
        self.data_window = None
        for data in datas:
            if len(data) < chunk_size:
                self.data_window = data
                return
            data_int16 = np.frombuffer(data, np.int16)
            data_float32 = self.int2float(data_int16)
            confidence = self.model(torch.from_numpy(data_float32), visitor.SAMPLE_RATE).item()
            if self.is_plot:
                self.confidences.append(confidence)
            if confidence>=0.2:
                self.exec_next(data)

    def stop(self, data):
        if self.is_plot:
            time = list(range(len(self.confidences)))
            plt.plot(time, self.confidences)
            plt.title('confidences')
            plt.xlabel('index')
            plt.ylabel('value')
            plt.show()
        self.stop_next(data)

    def int2float(self, data):
        abs_max = np.abs(data).max()
        data = data.astype('float32')
        if abs_max > 0:
            data *= 1 / 32768
        data = data.squeeze()
        return data
