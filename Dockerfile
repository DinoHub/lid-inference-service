# TODO: replace FROM
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get update && apt-get install -y gcc libsndfile1 ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN pip install --no-cache-dir torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV TRANSFORMERS_CACHE="/models/transformers_cache"

WORKDIR "/src"
CMD ["python", "app.py"]
