FROM python:3.11-slim

WORKDIR /app

# System deps for scikit-learn + scapy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app.py .

# Models and data are mounted via docker-compose volumes
# so we don't COPY them here — keeps image lean and lets
# you update models without rebuilding

EXPOSE 5000

CMD ["python", "app.py"]
