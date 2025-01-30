import numpy as np
from pydub import AudioSegment
import visitor


def louder(data, db):
    if not data:
        return data
    data_int16 = np.frombuffer(data, dtype=np.int16)
    audio_segment = AudioSegment(data_int16.tobytes(), frame_rate=visitor.SAMPLE_RATE, sample_width=2,
                                 channels=visitor.CHANNELS)
    audio_segment = audio_segment + db
    return audio_segment.raw_data


class LouderStreamVisitor(visitor.Visitor):
    db = 0

    def __init__(self, db=0):
        self.db = db

    def start(self, data):
        data = louder(data, self.db)
        self.start_next(data)

    def exec(self, data):
        data = louder(data, self.db)
        self.exec_next(data)

    def stop(self, data):
        data = louder(data, self.db)
        self.stop_next(data)
