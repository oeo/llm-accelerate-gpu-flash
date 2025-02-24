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
        nvidia-utils-535 && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chmod +x scripts/*.sh scripts/models/*.sh tests/*.sh

CMD ["./scripts/start-server.sh"] 