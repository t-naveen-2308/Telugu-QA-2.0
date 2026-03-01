# Telugu QA Backend

FastAPI backend for the Telugu Question Answering System.

## Features

- **QA Prediction** - Morphology-aware extractive QA
- **Model Selection** - MuRIL, mBERT, XLM-R, IndicBERT support
- **Transliteration** - ITRANS to Telugu conversion
- **Speech-to-Text** - Telugu audio transcription (Sarvam AI Saaras v3)
- **Text-to-Speech** - Telugu audio generation (Sarvam AI Bulbul v3)
- **Evaluation Metrics** - Model performance data

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/models` | List available models |
| POST | `/qa/predict` | Get answer for question |
| POST | `/transliterate` | ITRANS → Telugu |
| POST | `/tts` | Text to speech (MP3) |
| POST | `/asr/transcribe` | Speech to text |
| GET | `/examples` | Sample contexts |
| GET | `/training-data` | Training loss data |
| GET | `/health` | Health check |

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set environment variable for ASR/TTS:
```bash
# .env file
SARVAM_API_KEY=your_sarvam_api_key_here
```

3. Run the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Request/Response Examples

### QA Prediction

```bash
curl -X POST http://localhost:8000/qa/predict \
  -H "Content-Type: application/json" \
  -d '{
    "context": "హైదరాబాద్ తెలంగాణ రాజధాని.",
    "question": "తెలంగాణ రాజధాని ఏది?",
    "model_key": "muril"
  }'
```

Response:
```json
{
  "answer": "హైదరాబాద్",
  "confidence": 0.95,
  "original_answer": "హైదరాబాద్",
  "refinement_applied": false,
  "removed_suffixes": []
}
```

### Transliteration

```bash
curl -X POST http://localhost:8000/transliterate \
  -H "Content-Type: application/json" \
  -d '{"text": "namaste"}'
```

Response:
```json
{
  "original": "namaste",
  "telugu": "నమస్తే"
}
```
