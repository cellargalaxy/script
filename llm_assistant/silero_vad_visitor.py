import visitor
import numpy as np
import torch
import pyaudio
import matplotlib.pyplot as plt

is_speaking = True


class SileroVadStreamVisitor(visitor.Visitor):
    model = None
    buffer = None
    tmp = []

    def __init__(self):
        self.model, utils = torch.hub.load(repo_or_dir='./model/silero-vad/master', model='silero_vad', source='local')
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    def start(self, data):
        self.start_next(data)

    def exec(self, data):
        if self.buffer is None:
            self.buffer = data
        else:
            self.buffer += data

        chunk_size = visitor.CHUNK_SIZE
        match visitor.FORMAT:
            case pyaudio.paInt16:
                chunk_size *= 2
            case _:
                raise ValueError("该音频流格式未实现")

        datas = [self.buffer[i:i + chunk_size] for i in range(0, len(self.buffer), chunk_size)]
        self.buffer = None
        for data in datas:
            if len(data) < chunk_size:
                self.buffer = data
                return
            data_int16 = np.frombuffer(data, np.int16)
            data_float32 = self.int2float(data_int16)
            confidence = self.model(torch.from_numpy(data_float32), visitor.SAMPLE_RATE).item()
            # print('confidence', confidence)
            global is_speaking
            if confidence >= 0.5 and not is_speaking:
                is_speaking = True
                print('is_speaking', is_speaking, 'confidence', confidence)
            if confidence < 0.5 and is_speaking:
                is_speaking = False
                print('is_speaking', is_speaking, 'confidence', confidence)

            self.tmp.append(confidence)

            self.exec_next(data)

    def stop(self, data):
        time = list(range(len(self.tmp)))
        plt.plot(time, self.tmp)
        plt.title('按时间顺序排列的值')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.show()

        self.stop_next(data)

    def int2float(self, data):
        abs_max = np.abs(data).max()
        data = data.astype('float32')
        if abs_max > 0:
            data *= 1 / 32768
        data = data.squeeze()
        return data
