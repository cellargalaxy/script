from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

model, df_state, _ = init_df()
audio, _ = load_audio('gen_sub/output/demo/demucs/htdemucs/wav/vocals.wav', sr=df_state.sr())
enhanced = enhance(model, df_state, audio)
save_audio("enhanced_DeepFilterNet.wav", enhanced, df_state.sr())