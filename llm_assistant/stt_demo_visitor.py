import stt_visitor


class SttDemoStreamVisitor(stt_visitor.SttStreamVisitor):

    def __init__(self):
        super().__init__()

    def stt(self, data):
        print('stt', data)
