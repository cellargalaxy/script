python3 -m venv venv
source ./venv/bin/activate

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

#conda install pytorch torchvision torchaudio cpuonly -c pytorch
#conda install whisperx

#pip install sounddevice --proxy http://127.0.0.1:10808
#pip install rich --proxy http://127.0.0.1:10808
#pip install openai-whisper --proxy http://127.0.0.1:10808

#pip freeze > requirements.txt

#TMPDIR=./tmp pip install torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
#TMPDIR=./tmp pip install vosk -i https://pypi.tuna.tsinghua.edu.cn/simple