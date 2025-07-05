import paddle

paddle.set_device("cpu")
from paddlespeech.cli.vector import VectorExecutor
from paddlespeech.vector.cluster.diarization import (
    do_AHC,
    merge_ssegs_same_speaker,
    write_rttm
)


def diarize_to_rttm(audio_path, rttm_out, threshold=1.0, num_speakers=None):
    # 把 paddle 强制设为 CPU/GPU 根据你的环境
    paddle.set_device("cpu")  # 或 cpu

    # 1. 提取嵌入与语音段信息
    executor = VectorExecutor()
    emb_segments = executor(audio_file=audio_path)
    # emb_segments 是 List[EmbeddingMeta]，包含 time span 和嵌入

    # 2. 聚类：Agglomerative Hierarchical Clustering (AHC)
    cluster_labels = do_AHC(emb_segments)

    # 3. 合并同一说话人相邻段
    merged = merge_ssegs_same_speaker(emb_segments, cluster_labels)

    # 4. 写成 rttm 文件（标准 diarization 输出）
    write_rttm(merged, rttm_out)
    print(f"✅ 输出 RTTM 文件到 {rttm_out}")


if __name__ == "__main__":
    diarize_to_rttm("../aaa/output/long/segment_divide/arrange.wav", "meeting.rttm", threshold=1.2, num_speakers=2)
