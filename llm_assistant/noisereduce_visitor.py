import visitor
import math
import numpy as np
import noisereduce as nr


class NoiseReduceStreamVisitor(visitor.Visitor):
    window_data = None
    trim_count = 0
    trim_len = 0
    first = True

    def __init__(self):
        self.trim_count = math.ceil(visitor.NOISE_TRIM_SECONDS / (visitor.CHUNK_SIZE / visitor.SAMPLE_RATE))

    def start(self, data):
        self.first = True
        self.start_next(data)

    def exec(self, data):
        if self.window_data is None:
            self.window_data = data
        else:
            self.window_data += data

        self.trim_len = len(data) * self.trim_count
        if len(self.window_data) > self.trim_len * 2:
            audio_data = np.frombuffer(self.window_data, dtype=np.int16)
            reduced_noise = nr.reduce_noise(y=audio_data, sr=visitor.SAMPLE_RATE)
            reduced_noise = reduced_noise.astype(np.int16).tobytes()
            if self.first:
                self.exec_next(reduced_noise[:self.trim_len])
                self.first = False
            reduced_noise = reduced_noise[self.trim_len:]
            reduced_noise = reduced_noise[:-self.trim_len]
            self.exec_next(reduced_noise)
            self.window_data = self.window_data[len(reduced_noise):]

    def stop(self, data):
        if len(self.window_data) > 0:
            audio_data = np.frombuffer(self.window_data, dtype=np.int16)
            reduced_noise = nr.reduce_noise(y=audio_data, sr=visitor.SAMPLE_RATE)
            reduced_noise = reduced_noise.astype(np.int16).tobytes()
            if self.first:
                self.exec_next(reduced_noise[:self.trim_len])
                self.first = False
            reduced_noise = reduced_noise[self.trim_len:]
            self.exec_next(reduced_noise)
        self.stop_next(None)