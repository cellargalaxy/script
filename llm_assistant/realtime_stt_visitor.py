import visitor
import threading
import time
from RealtimeSTT import AudioToTextRecorder


class RealtimeSttStreamVisitor(visitor.Visitor):
    recorder = None
    data_window = None
    lock = None
    thread = None
    consuming = True

    def __init__(self, model):
        self.recorder = AudioToTextRecorder(model, device="cpu", compute_type="int8", realtime_model_type=model,
                                            use_microphone=False,spinner=False)
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.consumer)
        self.thread.start()

    def consumer(self):
        while True:
            print('data',self.consuming)
            if not self.consuming:
                return
            time.sleep(0.1)
            with self.lock:
                data = self.data_window
                self.data_window = None
            self.recognize(data)

    def start(self, data):
        self.accept(data)
        self.start_next(data)

    def exec(self, data):
        self.accept(data)
        self.exec_next(data)

    def stop(self, data):
        self.accept(data)
        self.consuming = False
        self.thread.join()
        self.stop_next(data)

    def accept(self, data):
        time.sleep(0.1)
        if not data:
            return
        with self.lock:
            if self.data_window is None:
                self.data_window = data
            else:
                self.data_window += data

    def recognize(self, data):
        if not data:
            return
        self.recorder.feed_audio(data)
        print("Transcription: ", self.recorder.text())
