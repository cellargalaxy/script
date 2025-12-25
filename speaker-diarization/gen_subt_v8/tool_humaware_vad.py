import numpy as np
import util
from pydub import AudioSegment
import torch
from pydub import AudioSegment
import numpy as np
import os
import util
from torch.hub import load_state_dict_from_url
from huggingface_hub import hf_hub_download

logger = util.get_logger()

model = None


def get_model():
    global model
    if model:
        return model
    model_path = hf_hub_download(repo_id="CuriousMonkey7/HumAware-VAD", filename="HumAwareVAD.jit")
    model = torch.jit.load(model_path)
    model.eval()
    return model


def pydub_humaware_vad(audio: AudioSegment):
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples())
    waveform = torch.from_numpy(samples).float() / 32768.0
    waveform = waveform.unsqueeze(0)
    return waveform


def vad_confidence(audio: AudioSegment):
    device = util.get_device_type()
    waveform = pydub_humaware_vad(audio)
    waveform = waveform.to(device)
    model = get_model()
    frame_size = 512
    hop_size = 512
    outputs = []
    for i in range(0, waveform.shape[1] - frame_size + 1, hop_size):
        frame = waveform[:, i:i + frame_size]
        out = model(frame, 16000)
        outputs.append(out)
    probs = torch.cat(outputs, dim=0)
    confidences = probs.squeeze(1).detach().cpu().tolist()
    expanded = [0.0] * len(audio)
    window_ms = 16000.0 / 512.0
    for i, conf in enumerate(confidences):
        start = int(i * window_ms)
        end = min(int((i + 1) * window_ms), len(audio))
        for ms in range(start, end):
            expanded[ms] = conf
    return expanded
