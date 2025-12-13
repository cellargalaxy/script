import util
from pyannote.audio import Inference
import numpy as np
import torch
from pyannote.audio import Model

logger = util.get_logger()

inference = None


def get_inference():
    global inference
    if inference:
        return inference
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(model, window="whole")
    inference.to(torch.device(util.get_device_type()))
    return inference


def pydub_pyannote(audio):
    samples = np.array(audio.get_array_of_samples())
    samples_float = samples.astype(np.float32) / audio.max_possible_amplitude
    waveform = torch.tensor(samples_float, dtype=torch.float32)
    waveform = waveform.unsqueeze(0)
    signal = {'waveform': waveform, 'sample_rate': audio.frame_rate}
    return signal


def extract_embedding(audio):
    signal = pydub_pyannote(audio)
    inference = get_inference()
    embedding = inference(signal)
    embedding = np.array(embedding).reshape(1, -1)
    embedding = np.squeeze(embedding)
    return embedding


def exec_gc():
    global inference
    inference = None
    util.exec_gc()
