FROM python:3.10-slim

WORKDIR /app

# System deps for torchaudio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only backend first (better caching)
COPY backend/ backend/

# Install deps
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose port (Railway uses $PORT)
EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "${PORT}", "--workers", "1"]
