import visitor
import threading
import time


class SttStreamVisitor(visitor.Visitor):
    data_window = None
    lock = None
    thread = None
    consuming = True

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.consumer)
        self.thread.start()

    def consumer(self):
        while self.consuming:
            time.sleep(0.5)
            with self.lock:
                data = self.data_window
                self.data_window = None
            if not data:
                continue
            self.stt(data)

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
        if not data:
            return
        with self.lock:
            if self.data_window is None:
                self.data_window = data
            else:
                self.data_window += data
