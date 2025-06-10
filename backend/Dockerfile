# backend.Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app/

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    python3-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
