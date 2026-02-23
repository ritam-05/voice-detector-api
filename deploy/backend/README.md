# Voice Detector API

Quick start:
- Install: pip install -r requirements.txt
- Ensure model checkpoints are available in ./models/
- Run locally: python app.py  (or `uvicorn app:app --reload`)
- POST /predict with a form file field named `file` (wav/mp3/flac/m4a)
- GET /health for a simple status check
