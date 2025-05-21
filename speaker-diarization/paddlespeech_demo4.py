from speechbrain.inference.diarization import Speech_Emotion_Diarization

diarization = Speech_Emotion_Diarization.from_hparams(source="speechbrain/emotion-diarization-wavlm-large", savedir="tmpdir")
frame_class=diarization.diarize_file("short.wav")
print(frame_class)
# print('est_sources', json.dumps(est_sources))
