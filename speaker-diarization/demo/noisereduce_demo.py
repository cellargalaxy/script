from scipy.io import wavfile
import noisereduce as nr
rate, data = wavfile.read('gen_sub/output/demo/demucs/htdemucs/wav/vocals.wav')
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)