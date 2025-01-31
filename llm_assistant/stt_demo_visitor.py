import stt_visitor


class SttDemoStreamVisitor(stt_visitor.SttStreamVisitor):

    def stt(self, data):
        print('stt', data)
