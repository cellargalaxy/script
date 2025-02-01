import visitor
import wave
import pyaudio
from faster_whisper import WhisperModel
import io
from pydub import AudioSegment
import threading
import time


class FasterWhisperStreamVisitor(visitor.Visitor):
    model = None
    lock = None
    thread = None
    consuming = True
    data_list = []
    last_data_time = 0
    sentence_window_len = 0
    sentences = []

    def __init__(self, model, sentence_window_len=3):
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.consumer)
        self.thread.start()
        self.sentence_window_len = sentence_window_len

    def consumer(self):
        while self.consuming:
            time.sleep(0.2)
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

        sentences = self.transcribe(data)

        if len(sentences) > self.sentence_window_len:
            print(f"\r", end="")
            for i in range(len(sentences) - self.sentence_window_len):
                print(f'done {sentences[i].text}')
        print(f"\rding {'；'.join([sentence.text for sentence in sentences[-self.sentence_window_len:]])}", end="")

        end_ms = -1
        for i in range(len(sentences) - self.sentence_window_len):
            end_ms = sentences[i].end * 1000
        if end_ms > 0:
            binary_io = self.gen_wave_binary_io(data)
            audio = AudioSegment.from_file(binary_io, format="wav")
            audio = audio[end_ms:]
            data = audio.raw_data
            with self.lock:
                data_list = [data]
                for i in range(data_list_len, len(self.data_list)):
                    data_list.append(self.data_list[i])
                self.data_list = data_list

        if self.is_mute():
            self.data_list = []

            print(f"\r", end="")
            for sentence in sentences[-self.sentence_window_len:]:
                print(f'done {sentence.text}')

    def transcribe(self, data):
        binary_io = self.gen_wave_binary_io(data)
        segments, info = self.model.transcribe(binary_io)
        sentences = []
        for segment in segments:
            sentences.append(segment)
        return sentences

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
        self.last_data_time = time.time()
        with self.lock:
            self.data_list.append(data)

    def is_mute(self):
        mute = time.time() - self.last_data_time >= visitor.MUTE_SECOND
        return mute
