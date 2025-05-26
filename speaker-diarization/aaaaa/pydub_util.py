from pydub import AudioSegment
import util


def cut_audio(audio_path, start_ms, end_ms, output_path):
    util.mkdir(output_path)
    audio = AudioSegment.from_wav(audio_path)
    cut = audio[start_ms:end_ms]
    cut.export(output_path, format="wav")
