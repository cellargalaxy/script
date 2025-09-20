import os


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['http_proxy'] = 'http://192.168.123.7:10808'
os.environ['https_proxy'] = 'http://192.168.123.7:10808'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1,192.168.123.7,mirrors.ustc.edu.cn,hf-mirror.com'

from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

model, df_state, _ = init_df()
audio, _ = load_audio("uvr_demo/output/noreverb.wav", sr=df_state.sr())
enhanced = enhance(model, df_state, audio)
save_audio("deepfilternet_demo.wav", enhanced, df_state.sr())