sudo apt install -y libc++1 # ten-vad需安装

pip install GPUtil==1.4.0
pip install inputimeout==1.0.4
pip install torch==2.8.0
pip install numpy==2.2.6
pip install pysubs2==1.8.0
pip install pydub==0.25.1
pip install -U --force-reinstall -v git+https://github.com/TEN-framework/ten-vad.git
pip install pyannote.audio==3.4.0
pip install --force-reinstall ctranslate2==4.4.0 # 最新版本仅支持CUDA12和cuDNN9，对于CUDA12和cuDNN8，需降级到4.4.0
pip install faster-whisper==1.2.0
pip install "audio-separator[gpu]"==0.39.0
pip install pyloudnorm==0.1.1