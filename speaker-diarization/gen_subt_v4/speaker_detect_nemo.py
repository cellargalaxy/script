import os
import util
import torch
import nemo.collections.asr as nemo_asr

logger = util.get_logger()

model = None
embedding_map = {}


def get_model():
    global model
    if model:
        return model
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    return model


def exec_gc():
    global model
    model = None
    global embedding_map
    embedding_map = None
    util.exec_gc()


def get_embedding(path, auth_token):
    global embedding_map
    embedding = embedding_map.get(path, None)
    if embedding is not None:
        return embedding
    model = get_model()
    embedding = model.get_embedding(path).squeeze()
    embedding_map[path] = embedding
    return embedding


def confidence_detect(path_i, path_j, auth_token):
    embedding_i = get_embedding(path_i, auth_token)
    embedding_j = get_embedding(path_j, auth_token)
    X = embedding_i / torch.linalg.norm(embedding_i)
    Y = embedding_j / torch.linalg.norm(embedding_j)
    similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
    similarity_score = (similarity_score + 1) / 2
    confidence = similarity_score.item()
    return confidence


def speaker_detect(audio_dir, auth_token):
    confidences = []
    files = util.listdir(audio_dir)
    files = [s for s in files if "speech.wav" in s]
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            path_i = os.path.join(audio_dir, files[i])
            path_j = os.path.join(audio_dir, files[j])
            confidence = confidence_detect(path_i, path_j, auth_token)
            confidences.append({"path_i": path_i, "path_j": path_j, "confidence": confidence})
    return confidences
