from RealtimeSTT import AudioToTextRecorder


def process_text(text):
    print(text)


if __name__ == '__main__':
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder("model/faster-whisper/base", device="cpu", compute_type="int8",
                                   realtime_model_type="model/faster-whisper/base",
                                            use_microphone=False)

    while True:
        recorder.text(process_text)
