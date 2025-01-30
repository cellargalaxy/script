import visitor
import pyaudio


class PyAudioVisitor(visitor.Visitor):
    running = False
    discard = 0

    def start(self, data):
        self.start_next(None)

    def exec(self, data):
        self.running = True
        self.discard = visitor.get_second_chunk_count(0.05)
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
        stream.close()
        p.terminate()

    def stop(self, data):
        self.running = False
        self.stop_next(None)
