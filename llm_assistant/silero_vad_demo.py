import numpy as np
import pyaudio
import torch


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()
    return sound


model, utils = torch.hub.load(repo_or_dir='model/silero-vad/master', model='silero_vad', source='local')
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
num_samples = 512

audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

for i in range(0, 1000):
    audio_chunk = stream.read(num_samples)
    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)
    new_confidence = model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()
    print('new_confidence', new_confidence)
