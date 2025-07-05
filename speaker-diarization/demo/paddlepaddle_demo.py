# 安装依赖
# pip install paddlespeech sklearn webrtcvad soundfile

import os
import numpy as np
import soundfile as sf
from paddlespeech.cli.vector import VectorExecutor
from paddlespeech.vector.cluster.diarization import pipeline as diar_pipeline


def diarize(audio_path, num_speakers=None):
    # 提取语音活动段
    executor = VectorExecutor()
    emb_segments = executor(audio_file=audio_path, task="spk")
    # emb_segments 是 List[EmbeddingMeta]，包含每个 segment 的嵌入与时间元
    # 传入 pipeline 进行聚类
    diar_result = diar_pipeline(emb_segments, num_speakers=num_speakers)

    # diar_result: list of (start, end, speaker_id)
    print("=== DIARIZATION RESULTS ===")
    for start, end, spk in diar_result:
        print(f"[{start:.2f}s – {end:.2f}s]: speaker_{spk}")
    return diar_result


if __name__ == "__main__":
    audio_path = "meeting.wav"  # 替换为你的文件
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"{audio_path}不存在")
    diarize(audio_path, num_speakers=2)  # 可选指定说话人数
