import os
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.speaker import SpeakerRecognition

# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
# signal, fs = torchaudio.load('output/demo/audio_split_segment_test/00004_speech.wav')
# embeddings = classifier.encode_batch(signal)
# print('embeddings', embeddings)

audio_dir = '../aaa/output/demo/audio_class/SPEAKER_01'
for file in os.listdir(audio_dir):
    audio_path = os.path.join(audio_dir, file)
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir="pretrained_models/spkrec-ecapa-voxceleb")
    score, prediction = verification.verify_files('../aaa/output/demo/audio_class/SPEAKER_01/00091.wav', audio_path)
    print(audio_path, score, prediction)
