import visitor
import wave
import pyaudio
from datetime import datetime


class WaveSaveVisitor(visitor.Visitor):
    file_name = ''
    file = None

    def __init__(self, file_name=''):
        self.file_name = file_name
        if not self.file_name:
            self.file_name = '{}.wav'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.file = wave.open(self.file_name, 'wb')
        self.file.setnchannels(visitor.CHANNELS)
        self.file.setsampwidth(pyaudio.get_sample_size(visitor.FORMAT))
        self.file.setframerate(visitor.SAMPLE_RATE)

    def start(self, data):
        self.writeframes(data)
        self.start_next(data)

    def exec(self, data):
        self.writeframes(data)
        self.exec_next(data)

    def stop(self, data):
        self.writeframes(data)
        if self.file:
            self.file.close()
        self.file = None
        self.stop_next(data)

    def writeframes(self, data):
        if self.file and data:
            self.file.writeframes(data)


class WaveSegmentSaveVisitor(visitor.Visitor):
    file_name = ''

    def __init__(self, file_name='wav.wav'):
        self.file_name = file_name

    def start(self, data):
        self.writeframes(data)
        self.start_next(data)

    def exec(self, data):
        self.writeframes(data)
        self.exec_next(data)

    def stop(self, data):
        self.writeframes(data)
        self.stop_next(data)

    def writeframes(self, data):
        writeframes(data, self.file_name)


def writeframes(data, file_name='wav.wav'):
    if not data:
        return
    file = wave.open('{}-{}'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_name), 'wb')
    file.setnchannels(visitor.CHANNELS)
    file.setsampwidth(pyaudio.get_sample_size(visitor.FORMAT))
    file.setframerate(visitor.SAMPLE_RATE)
    file.writeframes(data)
    file.close()
