FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set Work Directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=True
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Update, upgrade, install packages and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    pkg-config \
    zip \
    build-essential \
    software-properties-common \
    ffmpeg \
    libsndfile1 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY *.py ./
COPY *.json ./

STOPSIGNAL SIGINT

CMD ["python", "-u", "handler.py"]
