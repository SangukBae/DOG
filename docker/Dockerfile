FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 작업 디렉토리 구조 생성
RUN mkdir -p /workspace/outputs \
             /workspace/models \
             /workspace/preprocessed \
             /workspace/test_data \
             /workspace/model_data

VOLUME ["/workspace"]
