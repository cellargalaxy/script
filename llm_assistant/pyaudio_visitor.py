import visitor
import pyaudio
import time
import math


class PyAudioVisitor(visitor.Visitor):
    running = False
    discard = 0

    def start(self, data):
        self.start_next(None)

    def exec(self, data):
        self.running = True
        self.discard = math.ceil(0.05 / (visitor.CHUNK_SIZE / visitor.SAMPLE_RATE))
        p = pyaudio.PyAudio()
        stream = p.open(format=visitor.FORMAT, channels=visitor.CHANNELS, rate=visitor.SAMPLE_RATE, input=True,
                        frames_per_buffer=visitor.CHUNK_SIZE)
        while self.running:
            data = stream.read(visitor.CHUNK_SIZE)
            if self.discard > 0:
                self.discard -= 1
                continue
            self.exec_next(data)
        stream.stop_stream()
        time.sleep(0.1)
        stream.close()
        p.terminate()

    def stop(self, data):
        self.running = False
        self.discard = math.ceil(0.05 / (visitor.CHUNK_SIZE / visitor.SAMPLE_RATE))
        self.stop_next(None)
