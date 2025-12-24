import util
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np
import torch

logger = util.get_logger()

model = None


def get_model():
    global model
    if model:
        return model
    device = util.get_device_type()
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    return model


def exec_gc():
    global model
    model = None
    util.exec_gc()


def pydub_speechbrain(audio):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max
    signal = torch.tensor(samples).unsqueeze(0)
    return signal


def extract_embedding(audio):
    signal = pydub_speechbrain(audio)
    model = get_model()
    embedding = model.encode_batch(signal).squeeze().detach().cpu().numpy()
    return embedding
