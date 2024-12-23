FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    cmake \
    libthrift-dev \
    libboost-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: pyarrow
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY .env .
ENV PYTHONPATH=/app
CMD ["python", "app/run_evaluation.py"]