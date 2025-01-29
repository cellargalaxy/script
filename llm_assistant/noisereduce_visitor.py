import visitor
import math
import numpy as np
import noisereduce as nr
import pyaudio


class NoiseReduceStreamVisitor(visitor.Visitor):
    window_data = None
    trim_left_len = 0
    trim_right_len = 0
    first = True

    def __init__(self):
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
        self.trim_left_len = chunk_size * math.ceil(2 / (visitor.CHUNK_SIZE / visitor.SAMPLE_RATE))
        self.trim_right_len = chunk_size * math.ceil(0.2 / (visitor.CHUNK_SIZE / visitor.SAMPLE_RATE))

    def start(self, data):
        self.first = True
        self.start_next(data)

    def exec(self, data):
        if self.window_data is None:
            self.window_data = data
        else:
            self.window_data += data

        if len(self.window_data) > self.trim_left_len + self.trim_right_len:
            data_int16 = np.frombuffer(self.window_data, dtype=np.int16)
            reduced_noise = nr.reduce_noise(y=data_int16, sr=visitor.SAMPLE_RATE)
            reduced_noise = reduced_noise.astype(np.int16).tobytes()
            if self.first:
                self.exec_next(reduced_noise[:self.trim_left_len])
                self.first = False
            reduced_noise = reduced_noise[self.trim_left_len:]
            reduced_noise = reduced_noise[:-self.trim_right_len]
            self.exec_next(reduced_noise)
            self.window_data = self.window_data[len(reduced_noise):]

    def stop(self, data):
        if len(self.window_data) > 0:
            data_int16 = np.frombuffer(self.window_data, dtype=np.int16)
            reduced_noise = nr.reduce_noise(y=data_int16, sr=visitor.SAMPLE_RATE)
            reduced_noise = reduced_noise.astype(np.int16).tobytes()
            if self.first:
                self.exec_next(reduced_noise[:self.trim_left_len])
                self.first = False
            reduced_noise = reduced_noise[self.trim_left_len:]
            self.exec_next(reduced_noise)
        self.stop_next(None)
