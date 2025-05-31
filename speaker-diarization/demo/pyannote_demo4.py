from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model

model = Model.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token="")
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.1,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.1,
    # "collar": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)
vad = pipeline("../short.wav")
# `vad` is a pyannote.core.Annotation instance containing speech regions
print(vad)