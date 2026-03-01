"""
Telugu QA System - FastAPI Main Application

Serves the Telugu QA models via REST API.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import sys
from pathlib import Path
import io
import tempfile
import json
from models import (
    QARequest,
    QAResponse,
    TransliterateRequest,
    TransliterateResponse,
    TTSRequest,
    ModelInfo,
    EvaluationMetrics,
    TrainingData
)

# Global model cache
_engines = {}
_asr = None
_tts = None

def get_engine(model_key: str):
    """Get or load a QA engine."""
    global _engines
    if model_key not in _engines:
        from src.inference.qa_engine import load_qa_engine
        _engines[model_key] = load_qa_engine(model_key, use_morphology=True)
    return _engines[model_key]


def get_asr():
    """Get or load ASR engine."""
    global _asr
    if _asr is None:
        from src.speech import load_asr
        _asr = load_asr()
    return _asr


def get_tts():
    """Get or load TTS engine."""
    global _tts
    if _tts is None:
        from src.speech import load_tts
        _tts = load_tts()
    return _tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - preload default model."""
    print("🚀 Starting Telugu QA API...")
    try:
        # Preload MuRIL (best model)
        get_engine("muril")
        print("✅ MuRIL model loaded")
    except Exception as e:
        print(f"⚠️ Could not preload model: {e}")
    
    try:
        get_tts()
        print("✅ TTS engine loaded")
    except Exception as e:
        print(f"⚠️ TTS not available: {e}")
    
    yield
    
    print("👋 Shutting down Telugu QA API...")


# Create FastAPI app
app = FastAPI(
    title="Telugu QA API",
    description="Telugu Question Answering System with Speech Support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Telugu QA API",
        "version": "1.0.0",
        "endpoints": {
            "qa": "/qa/predict",
            "models": "/models",
            "transliterate": "/transliterate",
            "tts": "/tts",
            "asr": "/asr/transcribe"
        }
    }


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    """List available QA models with performance metrics."""
    models = [
        {
            "key": "muril",
            "name": "MuRIL",
            "description": "Best Quality - Multilingual Representations for Indian Languages",
            "exact_match": 68.53,
            "f1_score": 84.00,
            "recommended": True
        },
        {
            "key": "muril-domain",
            "name": "MuRIL-Domain",
            "description": "MuRIL + LoRA domain training (gov+lit+news) — F1 +42.6% on domain QA",
            "exact_match": 63.04,
            "f1_score": 66.80,
            "recommended": False
        },
        {
            "key": "mbert",
            "name": "mBERT",
            "description": "Multilingual BERT - Good balance of speed and accuracy",
            "exact_match": 61.14,
            "f1_score": 77.17,
            "recommended": False
        },
        {
            "key": "mbert-domain",
            "name": "mBERT-Domain",
            "description": "mBERT + LoRA domain training (gov+lit+news) — F1 +37.0%",
            "exact_match": 56.0,
            "f1_score": 73.2,
            "recommended": False
        },
        {
            "key": "xlmr",
            "name": "XLM-R",
            "description": "Cross-lingual Language Model - Strong cross-lingual transfer",
            "exact_match": 61.03,
            "f1_score": 77.18,
            "recommended": False
        },
        {
            "key": "indicbert",
            "name": "IndicBERT",
            "description": "Lightweight - Optimized for Indian languages",
            "exact_match": 9.82,
            "f1_score": 35.82,
            "recommended": False
        }
    ]
    return models


@app.post("/qa/predict", response_model=QAResponse)
async def predict(request: QARequest):
    """
    Get answer for a Telugu question from context.
    
    Uses morphology-aware refinement for improved answer boundaries.
    """
    try:
        engine = get_engine(request.model_key)
        
        # Use morphology-aware answering with rescoring
        result = engine.answer_with_rescore(request.question, request.context)
        
        return QAResponse(
            answer=result["answer"],
            confidence=result["score"],
            start=result["start"],
            end=result["end"],
            original_answer=result.get("original_answer", result["answer"]),
            original_confidence=result.get("original_score", result["score"]),
            refinement_applied=result.get("refinement_applied", False),
            removed_suffixes=result.get("removed_suffixes", []),
            confidence_improvement=result.get("confidence_improvement", 0.0),
            model_key=request.model_key,
            is_relevant=result.get("is_relevant", True)
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/transliterate", response_model=TransliterateResponse)
async def transliterate(request: TransliterateRequest):
    """
    Transliterate ITRANS/English text to Telugu.
    
    Uses Google Input Tools API for accurate transliteration and multiple options.
    """
    try:
        import requests
        url = f"https://www.google.com/inputtools/request?text={request.text}&ime=transliteration_en_te&num=5"
        response = requests.get(url)
        data = response.json()
        
        options = []
        if data[0] == "SUCCESS" and len(data[1]) > 0:
            word_options = [entry[1] for entry in data[1]]
            max_len = max((len(opts) for opts in word_options), default=0)
            
            for i in range(min(5, max_len)):
                combined = []
                for opts in word_options:
                    if i < len(opts):
                        combined.append(opts[i])
                    elif len(opts) > 0:
                        combined.append(opts[0])
                options.append(" ".join(combined))
        
        return TransliterateResponse(
            original=request.text,
            options=options
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transliteration error: {e}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert Telugu text to speech audio.
    
    Returns MP3 audio stream.
    """
    try:
        tts = get_tts()
        audio_bytes = tts.speak_bytes(request.text, slow=request.slow)
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")


@app.post("/asr/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe Telugu audio to text.
    
    Accepts: mp3, wav, m4a, ogg, webm
    """
    try:
        asr = get_asr()
        
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            result = asr.transcribe(tmp_path)
            return {
                "text": result["text"],
                "raw_text": result.get("raw_text", result["text"]),
                "language": result.get("language", "te"),
                "duration": result.get("duration"),
                "normalized": result.get("normalized", False)
            }
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")


@app.get("/metrics/{model_key}", response_model=EvaluationMetrics)
async def get_model_metrics(model_key: str):
    """Get evaluation metrics for a specific model."""
    eval_path = project_root / "data" / "processed" / f"evaluation_results_{model_key}.json"
    
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found for {model_key}")
    
    try:
        with open(eval_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", data)
        return EvaluationMetrics(
            model_key=model_key,
            exact_match=metrics.get("exact_match", 0),
            f1_score=metrics.get("f1", 0),
            total_samples=metrics.get("total", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {e}")


@app.get("/training-data", response_model=TrainingData)
async def get_training_data():
    """Get training loss data for all models."""
    return TrainingData(
        muril={
            "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000],
            "training_loss": [4.705, 3.330, 2.380, 2.125, 1.669, 1.587, 1.633, 1.580, 1.536, 1.264, 1.258, 1.250, 1.201, 1.205],
            "validation_loss": [4.694, 3.199, 2.203, 1.988, 1.934, 1.993, 1.969, 1.920, 1.855, 2.038, 2.011, 2.042, 2.065, 2.019],
            "best_step": 9000,
            "best_val_loss": 1.855
        },
        xlmr={
            "steps": [1000, 2000, 3000, 4000],
            "training_loss": [1.803, 1.338, 1.962, 2.016],
            "validation_loss": [2.337, 2.687, 2.255, 2.396],
            "best_step": 3000,
            "best_val_loss": 2.255
        },
        indicbert={
            "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000],
            "training_loss": [4.315, 3.506, 3.372, 3.176, 3.139, 2.898, 2.943, 2.920, 2.867, 2.815, 2.805, 2.550, 2.677, 2.589, 2.579, 2.549],
            "validation_loss": [4.401, 3.813, 3.590, 3.533, 3.598, 3.440, 3.416, 3.400, 3.418, 3.373, 3.340, 3.424, 3.350, 3.394, 3.350, 3.361],
            "best_step": 11000,
            "best_val_loss": 3.340
        },
        mbert={
            "steps": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000],
            "training_loss": [2.460, 2.147, 2.055, 1.941, 1.985, 1.795, 1.638, 1.603, 1.543, 1.638, 1.568, 1.507, 1.526, 1.204, 1.360, 1.257, 1.251, 1.151, 1.132],
            "validation_loss": [2.316, 2.041, 2.007, 1.944, 1.930, 1.860, 1.828, 1.787, 1.792, 1.805, 1.789, 1.850, 1.732, 1.882, 1.788, 1.823, 1.874, 1.869, 1.863],
            "best_step": 13000,
            "best_val_loss": 1.732
        },
        # Domain LoRA fine-tuning curves (5 epochs on 19k domain QA pairs)
        muril_domain={
            "steps": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 5955],
            "training_loss": [0.479, 0.279, 0.200, 0.164, 0.114, 0.105, 0.098, 0.088, 0.080, 0.078, 0.076, 0.074],
            "validation_loss": [0.368, 0.183, 0.121, 0.098, 0.070, 0.063, 0.058, 0.055, 0.049, 0.051, 0.050, 0.050],
            "best_step": 4500,
            "best_val_loss": 0.049
        },
        mbert_domain={
            "steps": [500, 1000, 1500, 2000, 2500, 3000, 3500],
            "training_loss": [0.634, 0.263, 0.254, 0.156, 0.101, 0.132, 0.117],
            "validation_loss": [0.449, 0.136, 0.087, 0.083, 0.081, 0.062, 0.057],
            "best_step": 3500,
            "best_val_loss": 0.057
        }
    )


@app.get("/examples")
async def get_examples():
    """Get example context-question pairs including domain-specific ones."""
    return {
        "తెలంగాణ": {
            "context": "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని. ఇది దక్కన్ పీఠభూమిపై ఉంది. హైదరాబాద్ జనాభా దాదాపు 1 కోటి. ఈ నగరంలో చార్మినార్, గోల్కొండ కోట వంటి చారిత్రక ప్రదేశాలు ఉన్నాయి.",
            "question": "తెలంగాణ రాజధాని ఏది?"
        },
        "భారతదేశం": {
            "context": "భారతదేశం దక్షిణ ఆసియాలో ఉన్న దేశం. భారతదేశ రాజధాని న్యూఢిల్లీ. భారతదేశ జనాభా 140 కోట్లు. భారతదేశంలో 28 రాష్ట్రాలు ఉన్నాయి.",
            "question": "భారతదేశ రాజధాని ఏది?"
        },
        "విజయవాడ": {
            "context": "విజయవాడ ఆంధ్రప్రదేశ్ లోని ముఖ్యమైన నగరం. ఇది కృష్ణా నది ఒడ్డున ఉంది. కనకదుర్గ ఆలయం ఈ నగరంలో ప్రసిద్ధ పుణ్యక్షేత్రం.",
            "question": "కనకదుర్గ ఆలయం ఎక్కడ ఉంది?"
        },
        "చార్మినార్": {
            "context": "చార్మినార్లో చాలా మంది పర్యాటకులు వస్తారు. చార్మినార్ హైదరాబాద్లో ఉంది. ఇది 1591లో నిర్మించబడింది.",
            "question": "చార్మినార్ ఎక్కడ ఉంది?"
        },
        "గాంధీ": {
            "context": "మహాత్మా గాంధీగారు భారత స్వాతంత్ర్య ఉద్యమానికి నాయకత్వం వహించారు. గాంధీగారితో పాటు అనేక మంది నాయకులు పోరాడారు. ఆయన అహింసా సిద్ధాంతాన్ని ప్రబోధించారు.",
            "question": "భారత స్వాతంత్ర్య ఉద్యమానికి ఎవరు నాయకత్వం వహించారు?"
        },
        # === Domain-specific examples (best with MuRIL-Domain / mBERT-Domain) ===
        "🏛️ రైతు బంధు పథకం": {
            "context": "వైద్య ఆరోగ్య శాఖ రైతు బంధు పథకాన్ని హైదరాబాద్ జిల్లాలో ప్రారంభించింది. ఈ పథకం ద్వారా అర్హులైన లబ్ధిదారులకు రూ.5,000 ఆర్థిక సహాయం అందజేయబడుతుంది.",
            "question": "రైతు బంధు పథకం ద్వారా ఎంత సహాయం అందుతుంది?"
        },
        "🏛️ ఆదాయ ధృవీకరణ": {
            "context": "ఆదాయ ధృవీకరణ పత్రం పొందడం కోసం హైదరాబాద్ జిల్లా మీసేవ కేంద్రంలో దరఖాస్తు చేసుకోవచ్చు. ఈ సర్టిఫికేట్ 15 రోజులు లో జారీ చేయబడుతుంది.",
            "question": "ఆదాయ ధృవీకరణ పత్రం ఎన్ని రోజుల్లో వస్తుంది?"
        },
        "📰 క్రికెట్ టెస్ట్ మ్యాచ్": {
            "context": "క్రికెట్: విరాట్ కోహ్లి అద్భుతమైన ప్రదర్శనతో సెంచరీ సాధించారు. ఈ బ్యాట్స్‌మన్ విశాఖపట్నంలో జరిగిన టెస్ట్ మ్యాచ్లో 156 రన్స్ తేడాతో గెలిచారు.",
            "question": "ఈ మ్యాచ్ ఎక్కడ జరిగింది?"
        },
        "📰 రాజకీయ సభ": {
            "context": "హైదరాబాద్లో భారతీయ జనతా పార్టీ ముఖ్యమంత్రి బహిరంగ సభ నిర్వహించారు. ఈ సందర్భంగా రాష్ట్ర అభివృద్ధిపై ప్రధానంగా మాట్లాడారు.",
            "question": "బహిరంగ సభ ఎక్కడ జరిగింది?"
        },
        "📚 వేమన పద్యం": {
            "context": "ఉప్పు కప్పురమ్ము ఒక్క పోలిక నుండు చూడ చూడ రుచుల జాడ లేరు పురుషులందు పుణ్య పురుషు లెఱుంగరు విశ్వదాభిరామ వినురవేమ. ఈ పద్యం వేమన రచించారు.",
            "question": "ఈ పద్యం రచయిత ఎవరు?"
        },
        "📚 పోతన కవి": {
            "context": "పోతన రచించిన ఆంధ్ర మహాభాగవతము తెలుగు సాహిత్యంలో అత్యంత ప్రసిద్ధ కావ్యం. ఈ గ్రంథంలో గజేంద్ర మోక్షం అనే ఘట్టం చాలా ప్రసిద్ధం.",
            "question": "ఆంధ్ర మహాభాగవతములో ప్రసిద్ధ ఘట్టం ఏది?"
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": list(_engines.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
