import visitor
import stt_visitor
import sherpa_ncnn
import numpy as np


class SherpaNcnnStreamVisitor(stt_visitor.SttStreamVisitor):
    recognizer = None
    text = ''

    def __init__(self):
        super().__init__()
        self.recognizer = sherpa_ncnn.Recognizer(
            tokens="model/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/tokens.txt",
            encoder_param="model/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.param",
            encoder_bin="model/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/encoder_jit_trace-pnnx.ncnn.bin",
            decoder_param="model/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.param",
            decoder_bin="model/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/decoder_jit_trace-pnnx.ncnn.bin",
            joiner_param="model/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.param",
            joiner_bin="model/sherpa-ncnn/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13/joiner_jit_trace-pnnx.ncnn.bin",
            num_threads=4,
            enable_endpoint_detection=True,
        )

    def stt(self, data):
        data_int16 = np.frombuffer(data, dtype=np.int16)
        data_float32 = data_int16.astype(np.float32) / 32768.0
        self.recognizer.accept_waveform(visitor.SAMPLE_RATE, data_float32)
        is_endpoint = self.recognizer.is_endpoint
        text = self.recognizer.text
        if is_endpoint:
            self.recognizer.reset()
        if text == self.text:
            return
        if is_endpoint:
            print(f"\r", end="")
            print(f'done {text}')
        else:
            print(f"\rding {text}", end="")
