import whisper
import numpy
import torch
from rich.console import Console

console = Console()
whisper_model = whisper.load_model("base.en")


class STT:
    def transcribe(self, audio_data) -> str:
        audio_np = (numpy.frombuffer(audio_data, dtype=numpy.int16).astype(numpy.float32) / 32768.0)
        if audio_np.size <= 0:
            console.print("[red]No audio recorded. Please ensure your microphone is working.")

        fp16 = torch.cuda.is_available()
        result = whisper_model.transcribe(audio_np, verbose=False, fp16=fp16)
        text = result["text"].strip()
        console.print(f"[yellow]You: {text}")
        return text
