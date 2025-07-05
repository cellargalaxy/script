import torch
import numpy as np
import os
import util
from pyannote.audio import Model
from scipy.spatial.distance import cdist
from pyannote.audio import Inference

logger = util.get_logger()

inference = None


def get_inference(auth_token):
    global inference
    if inference:
        return inference
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=auth_token)
    inference = Inference(model, window="whole")
    inference.to(torch.device(util.get_device_type()))
    return inference


def verify_confidence(file_path_i, file_path_j, auth_token):
    inference = get_inference(auth_token)
    embedding_i = inference(file_path_i)
    embedding_j = inference(file_path_j)
    embedding_i = np.array(embedding_i).reshape(1, -1)
    embedding_j = np.array(embedding_j).reshape(1, -1)
    distance = cdist(embedding_i, embedding_j, metric="cosine")[0, 0]
    confidence = 1 - distance
    return confidence


def verify_by_dir(file_dir, auth_token):
    confidences = []
    files = util.listdir(file_dir)
    files = [s for s in files if "speech" in s]
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            file_path_i = os.path.join(file_dir, files[i])
            file_path_j = os.path.join(file_dir, files[j])
            confidence = verify_confidence(file_path_i, file_path_j, auth_token)
            confidences.append({"file_path_i": file_path_i, "file_path_j": file_path_j, "confidence": confidence})
    return confidences
