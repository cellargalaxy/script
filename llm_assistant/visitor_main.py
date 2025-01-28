import pyaudio_visitor
import noisereduce_visitor
import silero_vad_visitor
import faster_whisper_visitor
import wave_save_visitor
import threading
import time

pyAudioVisitor = pyaudio_visitor.PyAudioVisitor()

noiseReduceStreamVisitor = noisereduce_visitor.NoiseReduceStreamVisitor()
noiseReduceStreamVisitor.set_pre(pyAudioVisitor)

# sileroVadStreamVisitor = silero_vad_visitor.SileroVadStreamVisitor()
# sileroVadStreamVisitor.set_pre(noiseReduceStreamVisitor)

fasterWhisperStreamVisitor = faster_whisper_visitor.FasterWhisperStreamVisitor()
fasterWhisperStreamVisitor.set_pre(noiseReduceStreamVisitor)

# waveSaveVisitor = wave_save_visitor.WaveSegmentSaveVisitor()
# waveSaveVisitor = wave_save_visitor.WaveSaveVisitor()
# waveSaveVisitor.set_pre(fasterWhisperStreamVisitor)


def exec():
    pyAudioVisitor.start(None)
    pyAudioVisitor.exec(None)


thread = threading.Thread(target=exec)
thread.start()

time.sleep(25)

pyAudioVisitor.stop(None)
thread.join()
