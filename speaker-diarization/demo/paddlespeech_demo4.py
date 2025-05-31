from speechbrain.inference.speaker import SpeakerRecognition
import  json

diarization = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir")
frame_class=diarization.diarize_file("short.wav")
print('frame_class', json.dumps(frame_class))
