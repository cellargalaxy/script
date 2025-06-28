import whisperx
import util

logger = util.get_logger()


def gen_subt(audio_path, auth_token):
    logger.info("生成字幕: %s", audio_path)

    device = util.get_device_type()
    compute_type = util.get_compute_type()

    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    aligned_result["language"] = result["language"]

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=auth_token, device=device)
    diarize_segments = diarize_model(audio)
    diarize_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
    diarize_result["language"] = result["language"]

    return diarize_result
