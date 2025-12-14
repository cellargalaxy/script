# https://pytorch.org/get-started/previous-versions/
# conda create -n v7_debug --clone v7_251213

conda env list
conda remove -n gen_subt_v7 --all
conda create -n gen_subt_v7 python=3.10
conda activate gen_subt_v7
conda deactivate

pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install "huggingface-hub<1.0.0"
pip install inputimeout==1.0.4
pip install "audio-separator[gpu]==0.39.1"
pip install pyloudnorm==0.1.1
sudo apt install -y libc++1 && pip install -U --force-reinstall -v git+https://github.com/TEN-framework/ten-vad.git
pip install pysubs2==1.8.0
pip install faster-whisper==1.2.1
pip install pyannote.audio==3.4.0
pip install speechbrain==1.0.3
pip install whisperx==3.7.4 #whisperx依赖torch==2.8.0，没法对torch降级

# requirements.txt没法直接安装torch，先手动安装torch再安装requirements.txt
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip freeze > requirements.txt


conda env list
conda remove -n audiosr --all
conda create -n audiosr python=3.10
conda activate audiosr
conda deactivate

pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install audiosr==0.0.7
pip install matplotlib==3.10.8
pip install pysubs2==1.8.0

# requirements.txt没法直接安装torch，先手动安装torch再安装requirements.txt
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_audiosr.txt
pip freeze > requirements_audiosr.txt