import os
import util
from speechbrain.inference.speaker import SpeakerRecognition

logger = util.get_logger()

verification = None


def get_verification():
    global verification
    if verification:
        return verification
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                   savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                                   run_opts={"device": "cuda"})
    return verification


def exec_gc():
    global inference
    inference = None
    util.exec_gc()


def confidence_detect(path_i, path_j):
    verification = get_verification()
    score, confidence = verification.verify_files(path_i, path_j)
    return score.item()


def speaker_detect(audio_dir, auth_token):
    confidences = []
    files = util.listdir(audio_dir)
    files = [s for s in files if "speech.wav" in s]
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            path_i = os.path.join(audio_dir, files[i])
            path_j = os.path.join(audio_dir, files[j])
            confidence = confidence_detect(path_i, path_j)
            confidences.append({"path_i": path_i, "path_j": path_j, "confidence": confidence})
    return confidences
