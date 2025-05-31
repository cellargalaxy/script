import whisperx
from whisperx.diarize import DiarizationPipeline, convert_whisperx_format
from whisperx.utils import SubtitlesProcessor

device = "cpu"
audio_file = "input.wav"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy) float16

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print('A-result["segments"]')
print(result["segments"])

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print('B-result["segments"]')
print(result["segments"])  # after alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="",
                                                     device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)


diarize_segments = convert_whisperx_format(diarize_segments)

result = whisperx.assign_word_speakers(diarize_segments, result)
print('C-diarize_segments')
print(diarize_segments)
print('D-result["segments"]')
print(result["segments"])  # segments are now assigned speaker IDs


def manual_merge_text_diarization(transcript_segments, diarization_segments):
    """
    transcript_segments: 来自 aligned_result["segments"]，每段含 start/end/text
    diarization_segments: 来自 DiarizationPipeline 输出，每段含 start/end/speaker
    """
    result = []

    for seg in transcript_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_text = seg["text"]

        # 找出所有在字幕时间段内发生重叠的说话人片段
        overlapping = [
            d for d in diarization_segments
            if not (d['end'] < seg_start or d['start'] > seg_end)
        ]

        # 如果有多个重叠，取时间交集最长的说话人
        if overlapping:
            speaker_durations = {}
            for d in overlapping:
                overlap_start = max(seg_start, d['start'])
                overlap_end = min(seg_end, d['end'])
                duration = overlap_end - overlap_start
                speaker = d['speaker']
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

            speaker = max(speaker_durations.items(), key=lambda x: x[1])[0]
        else:
            speaker = "unknown"

        result.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "speaker": speaker
        })

    return result


# 5. 合并说话人信息到转录结果中
final_result = manual_merge_text_diarization(result["segments"], diarize_segments)

# 6. 导出为字幕文件（SRT）
whisperx.utils.write_srt(final_result, "output.srt")
