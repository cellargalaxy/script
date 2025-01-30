import pyaudio
import math

CHUNK_SIZE = 512  # 设置音频流的数据块大小
FORMAT = pyaudio.paInt16  # 设置音频流的格式为16位整型，也就是2字节
CHANNELS = 1  # 设置音频流的通道数为1
SAMPLE_RATE = 16000  # 设置音频流的采样率为16KHz


class Visitor:
    next = None

    def set_next(self, next):
        self.next = next

    def set_pre(self, pre):
        if pre:
            pre.set_next(self)

    def start_next(self, data):
        if self.next:
            self.next.start(data)

    def exec_next(self, data):
        if self.next:
            self.next.exec(data)

    def stop_next(self, data):
        if self.next:
            self.next.stop(data)


def get_second_byte_size(second):
    chunk_size = get_chunk_byte_size()
    chunk_count = get_second_chunk_count(second)
    size = chunk_size * chunk_count
    return size


def get_second_chunk_count(second):
    count = math.ceil(second / (CHUNK_SIZE / SAMPLE_RATE))
    return count


def get_chunk_byte_size():
    match FORMAT:
        case pyaudio.paInt8:
            chunk_size = CHUNK_SIZE
        case pyaudio.paInt16:
            chunk_size = CHUNK_SIZE * 2
        case pyaudio.paInt24:
            chunk_size = CHUNK_SIZE * 3
        case pyaudio.paInt32:
            chunk_size = CHUNK_SIZE * 4
        case _:
            raise TypeError("未支持的音频流格式")
    return chunk_size
