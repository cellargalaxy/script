import visitor
import vosk
import json
import threading
import time


class VoskStreamVisitor(visitor.Visitor):
    recognizer = None
    data_window = None
    lock = None
    thread = None
    consuming = True

    def __init__(self):
        model = vosk.Model("model/vosk/vosk-model-cn-0.22")
        self.recognizer = vosk.KaldiRecognizer(model, visitor.SAMPLE_RATE)
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.consumer)
        self.thread.start()

    def consumer(self):
        while self.consuming:
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
        with self.lock:
            if self.data_window is None:
                self.data_window = data
            else:
                self.data_window += data

    def recognize(self, data):
        if not data:
            return
        if self.recognizer.AcceptWaveform(data):
            result = self.recognizer.Result()
            result_dict = json.loads(result)
            print(f"\r识别结果: {result_dict['text']}")
        else:
            partial_result = self.recognizer.PartialResult()
            partial_dict = json.loads(partial_result)
            print(f"\r部分识别: {partial_dict['partial']}", end="")
