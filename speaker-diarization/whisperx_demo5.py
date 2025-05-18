# 这个可用

import whisperx
import gc
import os
from whisperx.utils import get_writer
import torch



API_TOKEN =""
print(API_TOKEN)


# -----------------------------------------
# Parameters
# -----------------------------------------


device = "cpu"
audio_file =  "input.wav"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float32"  # change to "int8" if low on GPU mem (may reduce accuracy)
whisper_size = "large-v2"

# -----------------------------------------
# Model
# -----------------------------------------

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model(whisper_size, device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

# -----------------------------------------
# Transcription
# -----------------------------------------

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)


# # Save as an TXT file
# srt_writer = get_writer("txt", "captions/")
# srt_writer(result, audio_file, {})

# # Save as an SRT file
# srt_writer = get_writer("srt", "captions/")
# srt_writer(
#     result,
#     audio_file,
#     {"max_line_width": None, "max_line_count": None, "highlight_words": False},
# )

# # Save as a VTT file
# vtt_writer = get_writer("vtt", "captions/")
# vtt_writer(
#     result,
#     audio_file,
#     {"max_line_width": None, "max_line_count": None, "highlight_words": False},
# )

# # Save as a TSV file
# tsv_writer = get_writer("tsv", "captions/")
# tsv_writer(result, audio_file, {})

# # Save as a JSON file
# json_writer = get_writer("json", "captions/")
# json_writer(result, audio_file, {})


print("############### before alignment ###############")
print(result["segments"])  # before alignment

# delete model if low on GPU resources
import gc

gc.collect()
torch.cuda.empty_cache()
del model

# -----------------------------------------
# ???
# -----------------------------------------

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(
    language_code=result["language"], device=device
)
aligned_result = whisperx.align(
    result["segments"],
    model_a,
    metadata,
    audio,
    device,
    return_char_alignments=False,
)

print("############### after alignment ###############")
print(aligned_result)  # after alignment

# delete model if low on GPU resources
import gc

gc.collect()
torch.cuda.empty_cache()
del model_a


# -----------------------------------------
# Diarization
# -----------------------------------------

# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=API_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

diarize_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
# print(diarize_segments)

print("############### with speaker ID ###############")
print(diarize_result)  # segments are now assigned speaker IDs


# -----------------------------------------
# SAVE INTO FILES
# -----------------------------------------
# with open(f"captions/{file}_DIARIZED.json", "w") as f:
#     json.dump(result, f, indent=4)


# Save as a VTT file
diarize_result["language"] = result["language"]
vtt_writer = get_writer("vtt", ".")
vtt_writer(
    diarize_result,
    audio_file,
    {"max_line_width": None, "max_line_count": None, "highlight_words": False},
)