# https://pytorch.org/get-started/previous-versions/

conda env list
conda remove -n gen_subt_v7 --all
conda create -n gen_subt_v7 python=3.10
conda activate gen_subt_v7
conda deactivate

pip install "huggingface-hub<1.0.0"
pip install inputimeout
pip install --upgrade torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install "audio-separator[gpu]"
pip install pyloudnorm
sudo apt install -y libc++1 && pip install -U --force-reinstall -v git+https://github.com/TEN-framework/ten-vad.git
pip install pysubs2
pip install faster-whisper
pip install pyannote.audio==3.4.0
pip install speechbrain
pip install whisperx #whisperx依赖torch==2.8.0，没法对torch降级


conda env list
conda remove -n audiosr --all
conda create -n audiosr python=3.10
conda activate audiosr
conda deactivate

pip install --upgrade torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install audiosr


pip freeze > requirements.txt
pip install -r requirements.txt