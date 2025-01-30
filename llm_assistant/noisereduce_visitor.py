import visitor
import numpy as np
import noisereduce as nr


def reduce_noise(data):
    data_int16 = np.frombuffer(data, dtype=np.int16)
    reduced_noise = nr.reduce_noise(y=data_int16, sr=visitor.SAMPLE_RATE)
    reduced_noise = reduced_noise.astype(np.int16).tobytes()
    return reduced_noise


class NoiseReduceStreamVisitor(visitor.Visitor):
    window_data = None
    trim_left_len = 0
    trim_right_len = 0
    first = True

    def __init__(self):
        self.trim_left_len = visitor.get_second_byte_size(2)
        self.trim_right_len = visitor.get_second_byte_size(0.2)

    def start(self, data):
        self.first = True
        self.start_next(data)

    def exec(self, data):
        if self.window_data is None:
            self.window_data = data
        else:
            self.window_data += data

        if len(self.window_data) > self.trim_left_len + self.trim_right_len:
            reduced_noise = reduce_noise(self.window_data)
            if self.first:
                self.exec_next(reduced_noise[:self.trim_left_len])
                self.first = False
            reduced_noise = reduced_noise[self.trim_left_len:]
            reduced_noise = reduced_noise[:-self.trim_right_len]
            self.exec_next(reduced_noise)
            self.window_data = self.window_data[len(reduced_noise):]

    def stop(self, data):
        if len(self.window_data) > 0:
            reduced_noise = reduce_noise(self.window_data)
            if self.first:
                self.exec_next(reduced_noise[:self.trim_left_len])
                self.first = False
            reduced_noise = reduced_noise[self.trim_left_len:]
            self.exec_next(reduced_noise)
        self.stop_next(None)
