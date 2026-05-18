FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
