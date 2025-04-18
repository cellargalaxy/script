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
RUN #pip install --extra-index-url https://download.pytorch.org/whl/cpu "tflite_runtime" "opencv-python-headless" "torch" "torchvision" "torchaudio" "ultralytics"
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu "tflite_runtime" "opencv-python-headless" "ultralytics" "tf_keras" "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" "ai-edge-litert>=1.2.0" "onnx>=1.12.0" "onnx2tf" "onnxslim>=0.1.31" "tflite_support" "onnxruntime"

RUN bash download_model.sh
CMD ["python", "main.py"]
