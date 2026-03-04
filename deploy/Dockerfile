# ---- Base image ----
FROM python:3.10-slim

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Set working directory ----
WORKDIR /app

# ---- Copy backend code ----
COPY backend/ backend/

# ---- Install Python dependencies (FORCE CPU TORCH) ----
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r backend/requirements.txt

# ---- Expose port (HuggingFace spaces uses 7860, Render uses PORT env) ----
EXPOSE 7860

# ---- Start FastAPI (SINGLE WORKER ONLY) ----
CMD sh -c "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"
