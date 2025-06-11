# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies for Whisper, Transformers, and more
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional: pre-download HuggingFace and SentenceTransformer models to avoid delay on Render cold starts
RUN python3 -c "\
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer; \
from sentence_transformers import SentenceTransformer; \
import whisper; \
import spacy; \
from keybert import KeyBERT; \
pipeline('summarization', model='facebook/bart-large-cnn'); \
pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest'); \
pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', aggregation_strategy='simple'); \
pipeline('text-classification', model='unitary/toxic-bert'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
whisper.load_model('base'); \
spacy.cli.download('en_core_web_sm'); \
KeyBERT()"

# Copy app code
COPY . .

# Run app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
