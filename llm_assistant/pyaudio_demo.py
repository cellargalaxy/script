import wave
import pyaudio

CHUNK_SIZE = 512  # 设置音频流的数据块大小
FORMAT = pyaudio.paInt16  # 设置音频流的格式为16位整型，也就是2字节
CHANNELS = 1  # 设置音频流的通道数为1
SAMPLE_RATE = 16000  # 设置音频流的采样率为16KHz
RECORD_SECONDS = 20  # 设置录制时长

outfilepath = 'output.wav'
with wave.open(outfilepath, 'wb') as wf:
    p = pyaudio.PyAudio()
    # 设置wave文件的通道数
    wf.setnchannels(CHANNELS)
    # 设置wave文件的采样位数
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # 设置wave文件的采样率
    wf.setframerate(SAMPLE_RATE)

    # 打开音频流,input表示录音
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True)

    print('Recording...')
    # 循环写入音频数据
    for _ in range(0, SAMPLE_RATE // CHUNK_SIZE * RECORD_SECONDS):
        wf.writeframes(stream.read(CHUNK_SIZE))
    print('Done')

    stream.close()
    p.terminate()
