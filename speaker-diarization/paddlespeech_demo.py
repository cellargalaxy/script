from paddlespeech.cli.diar.infer import DiarExecutor

diar = DiarExecutor()
result = diar(input='208253969-7e35fe2a-7541-434a-ae91-8e919540555d.wav')
print(result)
