import nemo.collections.asr as nemo_asr
import torch
import numpy as np
import util

logger = util.get_logger()

model = None


def get_model():
    global model
    if model:
        return model
    model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
        model_name="nvidia/frame_vad_multilingual_marblenet_v2.0", strict=False)
    model = model.to(util.get_device_type())
    model.eval()
    return model


def pydub_nemo_vad(audio):
    samples = audio.get_array_of_samples()
    input_signal = np.array(samples).astype(np.float32) / audio.max_possible_amplitude
    input_signal = torch.tensor(input_signal).unsqueeze(0).float()
    input_signal_length = torch.tensor([input_signal.shape[1]]).long()
    return input_signal, input_signal_length


def vad_confidence(data):
    device = util.get_device_type()
    input_signal, input_signal_length = pydub_nemo_vad(data)
    model = get_model()
    with torch.no_grad():
        outputs = model(
            input_signal=input_signal.to(device),
            input_signal_length=input_signal_length.to(device)
        ).cpu()
    probs = torch.softmax(outputs, dim=-1)
    confidences = probs[:, :, 1]
    return confidences
