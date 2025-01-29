from scipy.io import wavfile
import noisereduce as nr
# load data
rate, data = wavfile.read("2025-01-29 15:39:05.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("mywav_reduced_noise.wav", rate, reduced_noise)