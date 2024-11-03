import stt_whisper
from rich.console import Console
import microphone_sounddevice

console = Console()
tts = stt_whisper.STT()
microphone = microphone_sounddevice.Microphone()

if __name__ == "__main__":
    try:
        while True:
            audio_data = microphone.record_audio()
            text = tts.transcribe(audio_data)
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")
    console.print("[blue]Session ended.")
