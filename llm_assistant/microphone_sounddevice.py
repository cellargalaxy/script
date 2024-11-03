import torch
import time
import threading
import sounddevice
from queue import Queue
from rich.console import Console

console = Console()


class Microphone:
    def __init__(self):
        console.print("Press Enter to start recording, then press Enter again to stop.")

    def record_audio(self):
        input()
        data_queue = Queue()
        stop_event = threading.Event()
        thread = threading.Thread(target=self.record_callback, args=(stop_event, data_queue))
        thread.start()
        input()
        stop_event.set()
        thread.join()
        return b"".join(list(data_queue.queue))

    def record_callback(self, stop_event, data_queue):
        def callback(indata, frames, time, status):
            if status:
                console.print(status)
            data_queue.put(bytes(indata))

        with sounddevice.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
            while not stop_event.is_set():
                time.sleep(0.1)
