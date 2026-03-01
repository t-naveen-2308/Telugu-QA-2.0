# Telugu Question Answering System (Telugu-QA 2.0)

A production-ready Telugu Question Answering system using fine-tuned transformer models with morphology-aware answer refinement.

## 📊 Model Performance

| Model | Parameters | Exact Match | F1 Score | Best For |
|-------|-----------|-------------|----------|----------|
| **MuRIL** | 237M | 68.53% | 84.00% | Best overall Telugu QA |
| **MuRIL-Domain** | 237M + LoRA | 70.2% | 85.1% | Domain-specific (govt/news/lit) |
| **mBERT** | 178M | 61.14% | 77.17% | Multilingual baseline |
| **XLM-R** | 270M | 61.03% | 77.18% | Cross-lingual tasks |
| **IndicBERT** | 33M | 9.82% | 35.82% | Lightweight/mobile |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (3.11 recommended)
- **Node.js 18+** (for frontend)
- **4GB+ RAM** (8GB recommended)
- **GPU optional** but recommended for faster inference

---

## 📦 Installation from ZIP

### Step 1: Extract the Archive

```bash
# Windows - Right-click → Extract All

# Linux/Mac
unzip telugu-qa-system.zip -d telugu-qa-system
cd telugu-qa-system
```

### Step 2: Create Python Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 5: Configure Environment

Create a `.env` file in the project root:

```env
# API Keys for Speech Services
SARVAM_API_KEY=sk_your-sarvam-key-here   # Best Telugu quality
OPENAI_API_KEY=sk_your-openai-key-here   # Good fallback

# Speech Provider Toggle (sarvam, openai, google)
# Automatic fallback: sarvam -> openai -> google
ASR_PROVIDER=sarvam
TTS_PROVIDER=sarvam

# Optional: Model settings
DEFAULT_MODEL=muril
DEVICE=cuda  # or "cpu"
```

---

## 🖥️ Running the Application

### Option A: Full Stack (Frontend + Backend)

**Terminal 1 - Start Backend:**
```bash
# From project root, with venv activated
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```

Access the application at: **http://localhost:5173**

### Option B: Backend Only (API)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API documentation: **http://localhost:8000/docs**

---

## 📁 Project Structure

```
telugu-qa-system/
├── backend/                      # FastAPI REST API
│   ├── src/                      # Core source code
│   │   ├── api/                  # API endpoints
│   │   ├── config/               # Configuration files
│   │   │   ├── model_config.yaml # Model settings
│   │   │   └── training_config.yaml # Training parameters
│   │   ├── data/                 # Data loading utilities
│   │   ├── inference/            # QA inference engine
│   │   ├── morphology/           # Telugu morphology processing
│   │   ├── speech/               # Speech components (multi-provider)
│   │   ├── training/             # Training utilities
│   │   └── utils/                # Helper functions
│   ├── main.py                   # App entrypoint
│   ├── models.py                 # Pydantic models
│   └── requirements.txt          # Backend-specific deps
│
├── frontend/                     # React + TypeScript UI
│   ├── src/
│   │   ├── App.tsx               # Main component
│   │   ├── api.ts                # API client
│   │   └── components/           # UI components
│   ├── package.json
│   └── vite.config.ts
│
├── data/                         # Datasets (Ignored in Git, download from Drive)
│
├── models/                       # Trained model weights and adapters (Ignored in Git)
│
├── notebooks/                    # Training notebooks (Google Colab)
│   ├── 01_training_colab.ipynb   # MuRIL & IndicBERT training
│   ├── 02_train_xlmr_colab.ipynb # XLM-R training
│   ├── 03_train_mbert_colab.ipynb# mBERT training
│   └── 04_domain_lora_training.ipynb # Domain adaptation with LoRA
│
├── scripts/                      # Utility scripts (see scripts/README.md)
│   ├── data_collection/          # Data scraping scripts
│   ├── data_prep/                # Data conversion & validation
│   ├── qa_generation/            # QA pair generation
│   ├── augmentation/             # Data augmentation
│   ├── export/                   # Package data for Colab
│   ├── evaluation/               # Model evaluation
│   └── testing/                  # Feature tests
│
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```
### 🔗 Links

- **Data** [View Dataset](https://drive.google.com/drive/folders/1iLgR6i3wjM4GQZKZOluIVTpvSgfi_n7J?usp=drive_link)

- **Models** [View Trained Models](https://drive.google.com/drive/folders/1lv3qNzY_er_H6O_8YbER1ORnwkScTr-x?usp=drive_link)
   
---

## 🎯 Features

### 1. Multi-Model QA
- Switch between MuRIL, mBERT, XLM-R, IndicBERT
- Domain-adapted models (government, news, literature)
- Real-time confidence scores

### 2. Morphology-Aware Answers
- Automatic suffix removal (లో, గారు, తో, etc.)
- Question-type aware refinement
- Preserves meaningful endings (names, units)

### 3. Telugu Input Methods
- **Direct typing** - Use system Telugu keyboard
- **Transliteration** - Type English (ITRANS) → Telugu
  - `namaste` → `నమస్తే`
  - `hyderabad` → `హైదరాబాద్`

### 4. Speech Integration (Multi-Provider)
- **Voice input** - ASR with automatic fallback:
  - Sarvam AI Saaras v3 (best Telugu)
  - OpenAI Whisper (good multilingual)
  - Google Speech Recognition (free)
- **Voice output** - TTS with automatic fallback:
  - Sarvam AI Bulbul v3 (best Telugu)
  - OpenAI TTS
  - Google gTTS (free)

### 5. REST API
- OpenAPI/Swagger documentation
- Health checks and model info endpoints
- Batch processing support

---

## 🔧 API Reference

### Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/qa/predict` | POST | Answer a question |
| `/transliterate` | POST | ITRANS → Telugu |
| `/asr/transcribe` | POST | Speech to text |
| `/tts` | POST | Text to speech |
| `/examples` | GET | Get example QA pairs |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/qa/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question": "తెలంగాణ రాజధాని ఏది?",
    "context": "హైదరాబాద్ తెలంగాణ రాష్ట్ర రాజధాని. ఇది దక్కన్ పీఠభూమిపై ఉంది.",
    "model_name": "muril"
  }'
```

---

## 🏋️ Training Your Own Models

### What You Need to Upload to Colab

Create a zip file with:

```
colab-upload.zip
├── data/
│   └── processed/
│       ├── tequad_train.json      # Required: Training data
│       └── tequad_validation.json # Required: Validation data
└── config/
    └── training_config.yaml       # Optional: Custom config
```

**Generate the zip automatically:**
```bash
python scripts/export/export_for_colab.py
# Creates: colab-upload.zip
```

### Training Order

1. **Base model training** (choose one):
   - `01_training_colab.ipynb` → MuRIL + IndicBERT
   - `02_train_xlmr_colab.ipynb` → XLM-R
   - `03_train_mbert_colab.ipynb` → mBERT

2. **Domain adaptation** (optional, after base training):
   - `04_domain_lora_training.ipynb` → LoRA fine-tuning

### After Training

Download these files from Colab:
- `model.safetensors` (or `pytorch_model.bin`)
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

Place them in: `models/checkpoints/<model_name>/`

---

## 📝 Adding Custom Domain Data

### Step 1: Prepare Your Data

Create JSON files in SQuAD format:

```json
{
  "version": "1.0",
  "data": [
    {
      "title": "Article Title",
      "paragraphs": [
        {
          "context": "Your Telugu text here...",
          "qas": [
            {
              "id": "unique_id_001",
              "question": "మీ ప్రశ్న?",
              "answers": [
                {
                  "text": "సమాధానం",
                  "answer_start": 10
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

### Step 2: Place Data Files

```
data/domain/
├── your_domain/
│   ├── train.json       # Training data (SQuAD format)
│   └── validation.json  # Validation data
```

### Step 3: Verify Data Format

```bash
python scripts/data_prep/verify_squad.py data/domain/your_domain/train.json
```

### Step 4: Export for Colab

```bash
python scripts/export/export_domain_for_colab.py --domain your_domain
# Creates: domain-your_domain-colab.zip
```

### Step 5: Train Domain Adapter

Upload to Colab and run `04_domain_lora_training.ipynb`

---

## 📜 Script Reference

See [scripts/README.md](scripts/README.md) for detailed documentation.

### Data Preparation

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/data_prep/convert_to_squad.py` | Convert raw data to SQuAD format | `python scripts/data_prep/convert_to_squad.py input.json output.json` |
| `scripts/data_prep/verify_squad.py` | Validate SQuAD format | `python scripts/data_prep/verify_squad.py data.json` |
| `scripts/data_collection/scrape_*.py` | Scrape data from sources | `python scripts/data_collection/scrape_news.py` |
| `scripts/qa_generation/generate_qa_pairs.py` | Generate QA from text | `python scripts/qa_generation/generate_qa_pairs.py` |

### Training & Export

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/export/export_for_colab.py` | Package base training data | `python scripts/export/export_for_colab.py` |
| `scripts/export/export_domain_for_colab.py` | Package domain data | `python scripts/export/export_domain_for_colab.py --domain news` |

### Evaluation

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/evaluation/evaluate_model.py` | Evaluate single model | `python scripts/evaluation/evaluate_model.py --model muril` |
| `scripts/evaluation/evaluate_domain_models.py` | Evaluate domain models | `python scripts/evaluation/evaluate_domain_models.py` |
| `scripts/evaluation/run_evaluation.py` | Full evaluation suite | `python scripts/evaluation/run_evaluation.py` |

---

## 🐛 Troubleshooting

### "Model not found" Error

Ensure model checkpoints exist:
```
models/checkpoints/muril/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### Out of Memory Error

1. Use smaller model: `indicbert`
2. Reduce batch size in config
3. Use CPU: set `DEVICE=cpu` in `.env`

### Telugu Text Not Displaying

1. Install Telugu font (Noto Sans Telugu)
2. Windows: Settings → Time & Language → Language → Add Telugu

### Frontend Not Connecting to Backend

1. Ensure backend is running on port 8000
2. Check CORS settings in `backend/main.py`
3. Verify frontend API URL in `frontend/src/api.ts`

### Speech Features Not Working

1. Check provider setting in `.env` (`ASR_PROVIDER`, `TTS_PROVIDER`)
2. For Sarvam: Set `SARVAM_API_KEY` (https://dashboard.sarvam.ai/)
3. For OpenAI: Set `OPENAI_API_KEY` (https://platform.openai.com/)
4. For Google: No API key needed (free fallback)
5. To switch provider at runtime: Change `ASR_PROVIDER`/`TTS_PROVIDER` in `.env`
6. Ensure microphone permissions granted
7. Supported audio formats: WAV, MP3, WebM

---

## 📚 Technical Details

### Dataset

- **TeQuAD**: Telugu Question Answering Dataset
- Training: ~75,000 examples
- Validation: ~3,000 examples
- Test: ~950 examples

### Morphology Processing

The system includes Telugu-specific post-processing:

| Feature | Description |
|---------|-------------|
| Suffix removal | Removes grammatical suffixes (లో, గారు, తో) |
| Question-aware | Different rules for who/where/when questions |
| Compound handling | Normalizes compound words |
| Coreference | Resolves demonstratives (ఈ, ఆ, ఇది) |

### Model Architecture

| Model | HuggingFace ID | Layers | Hidden |
|-------|---------------|--------|--------|
| MuRIL | `google/muril-base-cased` | 12 | 768 |
| mBERT | `bert-base-multilingual-cased` | 12 | 768 |
| XLM-R | `xlm-roberta-base` | 12 | 768 |
| IndicBERT | `ai4bharat/indic-bert` | 12 | 768 |

---

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- [TeQuAD Dataset](https://github.com/AkhilMittapalli/TeQuAD)
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [Google MuRIL](https://huggingface.co/google/muril-base-cased)
- [AI4Bharat IndicBERT](https://huggingface.co/ai4bharat/indic-bert)
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)
