from ten_vad import TenVad
import scipy.io.wavfile as Wavfile

if __name__ == "__main__":
    sr, data = Wavfile.read('../demo_eng_single.wav')
    hop_size = int(sr / 10)
    threshold = 0.5
    ten_vad_instance = TenVad(hop_size, threshold)  # Create a TenVad instance
    num_frames = data.shape[0] // hop_size  # 帧数
    for i in range(num_frames):
        audio_data = data[i * hop_size: (i + 1) * hop_size]
        out_probability, out_flag = ten_vad_instance.process(audio_data)
        print("[%d] %0.6f, %d" % (i, out_probability, out_flag))
