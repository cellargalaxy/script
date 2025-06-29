import tempfile
import os
import util
import math
from pydub import AudioSegment
import util_subt
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

logger = util.get_logger()


def speaker_detect(audio_path, auth_token, min_speech_duration_ms=250):
    logger.info("说话人检测: %s", audio_path)

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    with tempfile.TemporaryDirectory() as tmp_dir:
        manifest_filepath = os.path.join(tmp_dir, 'input_manifest.json')
        output_dir = os.path.join(tmp_dir, 'output')
        rttm_path = os.path.join(tmp_dir, f"output/pred_rttms/{util.get_file_name(audio_path)}.rttm")
        meta = {
            'audio_filepath': audio_path,
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': None,
            'rttm_filepath': None,
            'uem_filepath': None
        }
        util.save_as_json(meta, manifest_filepath)

        config = OmegaConf.load("nemo_config.yaml")
        config.diarizer.manifest_filepath = manifest_filepath
        config.diarizer.out_dir = output_dir
        model = ClusteringDiarizer(cfg=config)
        model.diarize()
        results = rttm2segments(rttm_path)

    segments = []
    for i, segment in enumerate(results):
        if segment['start'] < 0:
            segment['start'] = 0
        if last_end < segment['end']:
            segment['end'] = last_end
        if segment['end'] - segment['start'] < min_speech_duration_ms:
            continue
        segments.append(segment)

    segments = util_subt.fix_overlap_segments(segments)

    del audio
    util.exec_gc()

    return segments


def rttm2segments(file_path):
    segments = []
    with open(file_path, 'r') as file:
        for line in file:
            segment = rttm2segment(line)
            if segment:
                segments.append(segment)
    return segments


def rttm2segment(line):
    texts = []
    for text in line.split(' '):
        text = text.strip()
        if text:
            texts.append(text)
    if len(texts) < 1:
        return None
    start = math.floor(float(texts[3]) * 1000)
    end = math.ceil((float(texts[3]) + float(texts[3])) * 1000)
    segment = {"start": start, "end": end, "speaker": texts[7]}
    return segment
