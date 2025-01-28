import visitor
import pyaudio


class PyAudioVisitor(visitor.Visitor):
    running = False

    def start(self, data):
        self.running = True
        self.start_next(None)

    def exec(self, data):
        self.running = True
        p = pyaudio.PyAudio()
        stream = p.open(format=visitor.FORMAT, channels=visitor.CHANNELS, rate=visitor.SAMPLE_RATE, input=True)
        while self.running:
            data = stream.read(visitor.CHUNK_SIZE)
            self.exec_next(data)
        stream.close()
        p.terminate()

    def stop(self, data):
        self.running = False
        self.stop_next(None)
