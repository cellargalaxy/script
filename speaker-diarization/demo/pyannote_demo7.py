from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
import pysubs2


def save_segments_as_srt(segments, file_path):
    results = []
    for i, segment in enumerate(segments):
        start = segment['start']
        end = segment['end']
        text = segment.get('text', '')
        if not text:
            text = f"[{segment.get('type', '')}|{segment.get('speaker', '')}] {segment['start']}->{segment['end']}"
        obj = {'start': start, 'end': end, 'text': text}
        results.append(obj)
    subs = pysubs2.load_from_whisper(results)
    subs.save(file_path)


model = Model.from_pretrained("pyannote/segmentation", use_auth_token="")
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
    "onset": 0.95, "offset": 0.95,
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0,
    # "collar": 0.0,
}
pipeline.instantiate(HYPER_PARAMETERS)
vad = pipeline("../demo_jpn.wav")
# `vad` is a pyannote.core.Annotation instance containing speech regions
print(vad)

segments = []
for segment, _, label in vad.itertracks(yield_label=True):
    print(f"Start: {segment.start:.2f}s, End: {segment.end:.2f}s, Label: {label}")
    segments.append({"start": segment.start, "end": segment.end, "type": label})
save_segments_as_srt(segments, 'pyannote_demo7.srt')
