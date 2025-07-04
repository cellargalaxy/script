import util
import util_subt
from pydub import AudioSegment
from ten_vad import TenVad
import scipy.io.wavfile as Wavfile
import math

logger = util.get_logger()


def activity_detect(audio_path,
                   frame_rate=10,
                   threshold=0.4,
                   speech_threshold=0.8,
                   silene_threshold=0.5,
                   min_speech_duration_ms=250):
    logger.info("活动检测: %s", audio_path)

    audio = AudioSegment.from_wav(audio_path)
    last_end = len(audio)

    simple_rate, data = Wavfile.read(audio_path)
    frame_simple = int(simple_rate / frame_rate)  # 每侦采样数
    frame_cnt = data.shape[0] // frame_simple  # 总帧数
    ten_vad_instance = TenVad(frame_simple)
    segments = []
    for i in range(frame_cnt):
        audio_data = data[i * frame_simple: (i + 1) * frame_simple]
        probability, _ = ten_vad_instance.process(audio_data)
        vad_type = 'silene'
        if probability >= threshold:
            vad_type = 'speech'
        pre_end = 0
        if len(segments) > 0:
            pre_end = segments[len(segments) - 1]['end']
        end = math.ceil((i + 1) * (1.0 / frame_rate) * 1000)
        if last_end < end:
            end = last_end
        if len(segments) == 0:
            segments.append({"start": pre_end, "end": end, "vad_type": vad_type, "vad_probabilities": [probability]})
            continue
        if segments[len(segments) - 1]['vad_type'] == vad_type:
            segments[len(segments) - 1]['end'] = end
            segments[len(segments) - 1]['vad_probabilities'].append(probability)
            continue
        segments.append({"start": pre_end, "end": end, "vad_type": vad_type, "vad_probabilities": [probability]})
    for i, segment in enumerate(segments):
        segments[i]['vad_probability'] = sum(segments[i]['vad_probabilities']) / len(segments[i]['vad_probabilities'])
    for i, segment in enumerate(segments):
        if speech_threshold <= segments[i]['vad_probability']:
            segments[i]['vad_type'] = 'speech'
            continue
        if segments[i]['vad_probability'] <= silene_threshold:
            segments[i]['vad_type'] = 'silene'
            continue
        if segments[i]['vad_type'] == 'speech' and segments[i]['end'] - segments[i]['start'] < min_speech_duration_ms:
            segments[i]['vad_type'] = 'silene'

    pre_end = 0
    if len(segments) > 0:
        pre_end = segments[len(segments) - 1]['end']
    if pre_end < last_end:
        segments.append({"start": pre_end, "end": last_end, "vad_type": 'silene'})

    segments = util_subt.fix_overlap_segments(segments)
    segments = util_subt.unit_segments(segments, 'vad_type')
    util_subt.check_coherent_segments(segments)

    del audio
    del data
    del ten_vad_instance
    util.exec_gc()

    return segments
