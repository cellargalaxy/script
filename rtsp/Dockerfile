FROM python:3.9-slim

WORKDIR /
COPY snapshot.sh .
COPY recording.sh .
COPY clean.sh .
COPY requirements.txt .
COPY download_model.sh .
COPY main.py .

RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bullseye-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bullseye-security main contrib non-free" >> /etc/apt/sources.list

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi-dev \
    ffmpeg \
    curl \
    bash \
    tzdata && \
    cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    tflite_runtime \
    opencv-python-headless \
    torch torchvision torchaudio \
    ultralytics \

RUN bash download_model.sh
CMD ["python", "main.py"]
