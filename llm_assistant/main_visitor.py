import pyaudio_visitor
import louder_visitor
import noisereduce_visitor
import silero_vad_visitor
import vosk_visitor
import wave_save_visitor
import threading
import time

visitors = []

visitors.append(pyaudio_visitor.PyAudioVisitor())
visitors.append(noisereduce_visitor.NoiseReduceStreamVisitor())
visitors.append(louder_visitor.LouderStreamVisitor(10))
visitors.append(silero_vad_visitor.SileroVadStreamFilterMuteVisitor())
visitors.append(vosk_visitor.VoskStreamVisitor())
visitors.append(wave_save_visitor.WaveSaveVisitor())

if not visitors:
    exit(0)
for i in range(len(visitors)):
    if i == 0:
        continue
    visitors[i].set_pre(visitors[i - 1])


def exec():
    visitors[0].start(None)
    visitors[0].exec(None)


thread = threading.Thread(target=exec)
thread.start()

time.sleep(250)

visitors[0].stop(None)
thread.join()
