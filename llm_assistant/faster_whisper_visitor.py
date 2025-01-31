import visitor
import wave
import pyaudio
from faster_whisper import WhisperModel
import io
from pydub import AudioSegment
import threading
import time
import wave_save_visitor


class FasterWhisperStreamVisitor(visitor.Visitor):
    model = None
    lock = None
    thread = None
    consuming = True
    data_list = []

    def __init__(self, model):
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.consumer)
        self.thread.start()

    def consumer(self):
        while self.consuming:
            time.sleep(3)
            self.stt()

    def stt(self):
        with self.lock:
            data_list_len = len(self.data_list)
        data = None
        for i in range(data_list_len):
            if data is None:
                data = self.data_list[i]
            else:
                data += self.data_list[i]
        if not data:
            return

        texts = self.transcribe(data)

        if len(texts) <= 2:
            for text in texts:
                print(text.text)
            return

        # binary_io = self.wave(self.text_data)
        # audio = AudioSegment.from_file(binary_io, format="wav")
        # for text in texts:
        #     start_ms = text.start * 1000
        #     audio = audio[start_ms:]
        #     self.text_data = audio.raw_data
        #     break

    def transcribe(self, data):
        print('---')
        print(len(data))
        wave_save_visitor.writeframes(data)

        binary_io = self.gen_wave_binary_io(data)
        segments, _ = self.model.transcribe(binary_io)
        texts = []
        for segment in segments:
            texts.append(segment)
        return texts

    def gen_wave_binary_io(self, data):
        binary_io = io.BytesIO()
        with wave.open(binary_io, 'wb') as wf:
            wf.setnchannels(visitor.CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(visitor.FORMAT))
            wf.setframerate(visitor.SAMPLE_RATE)
            wf.writeframes(data)
        binary_io.seek(0)
        return binary_io

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
            self.data_list.append(data)
