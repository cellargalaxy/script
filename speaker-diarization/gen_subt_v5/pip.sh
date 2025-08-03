sudo apt install -y libc++1

pip install GPUtil
pip install inputimeout
pip install torch
pip install numpy
pip install pysubs2
pip install pydub
pip install -U --force-reinstall -v git+https://github.com/TEN-framework/ten-vad.git
pip install pyannote.audio
pip install --force-reinstall ctranslate2==4.4.0 # 注意 ： ctranslate2 的最新版本仅支持 CUDA 12 和 cuDNN 9。对于 CUDA 12 和 cuDNN 8，降级到 ctranslate2 的 4.4.0 版本
pip install faster-whisper
pip install "audio-separator[gpu]"
