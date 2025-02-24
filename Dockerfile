FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        numactl \
        curl \
        jq \
        bc \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libfreetype6-dev \
        liblcms2-dev \
        libwebp-dev \
        zlib1g-dev \
        gcc \
        g++ \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chmod +x scripts/*.sh scripts/models/*.sh tests/*.sh

CMD ["./scripts/start-server.sh"] 