FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir scapy==2.5.0

COPY capture.py .

# Needs NET_ADMIN + NET_RAW — set in docker-compose, not here
CMD ["python", "-u", "capture.py"]
