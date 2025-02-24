FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

WORKDIR /app

# Install system dependencies and NVIDIA utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        numactl \
        curl \
        jq \
        bc \
        gnupg2 \
        software-properties-common && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-drivers-535 \
        nvidia-utils-535 \
        cuda-toolkit-12-1 && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.1-1_all.deb

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chmod +x scripts/*.sh scripts/models/*.sh tests/*.sh

CMD ["./scripts/start-server.sh"] 