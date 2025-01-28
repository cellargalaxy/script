import pyaudio

CHUNK_SIZE = 512  # 设置音频流的数据块大小
FORMAT = pyaudio.paInt16  # 设置音频流的格式为16位整型，也就是2字节
CHANNELS = 1  # 设置音频流的通道数为1
SAMPLE_RATE = 16000  # 设置音频流的采样率为16KHz
NOISE_TRIM_SECONDS = 0.64  # 大约0.64秒是极限了
WHISPER_SECONDS = 5


class Visitor:
    next = None

    def set_next(self, next):
        self.next = next

    def set_pre(self, pre):
        if pre is None:
            return
        pre.set_next(self)

    def start_next(self, data):
        if self.next is not None:
            self.next.start(data)

    def exec_next(self, data):
        if self.next is not None:
            self.next.exec(data)

    def stop_next(self, data):
        if self.next is not None:
            self.next.stop(data)
