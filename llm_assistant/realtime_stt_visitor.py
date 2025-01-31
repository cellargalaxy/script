import stt_visitor
from RealtimeSTT import AudioToTextRecorder


class RealtimeSttStreamVisitor(stt_visitor.SttStreamVisitor):
    recorder = None

    def __init__(self, model):
        super().__init__()
        self.recorder = AudioToTextRecorder(model, device="cpu", compute_type="int8", realtime_model_type=model,
                                            use_microphone=False, spinner=False)

    def stt(self, data):
        print('stt', data)
        if not data:
            return
        self.recorder.feed_audio(data)
        print('done')
        print("Transcription: ", self.recorder.text())
