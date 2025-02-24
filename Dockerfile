FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    numactl \
    curl \
    jq \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh scripts/models/*.sh tests/*.sh

# Default command
CMD ["./scripts/start-server.sh"] 