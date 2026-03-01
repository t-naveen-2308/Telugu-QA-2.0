# Telugu QA Frontend

Modern React + TypeScript frontend for the Telugu Question Answering System.

## Features

- **Model Selection** - Switch between MuRIL, mBERT, XLM-R, and IndicBERT
- **Text Input** - Context + Question input with file upload support
- **Voice Input** - Record or upload audio for speech-to-text questions
- **Telugu Keyboard** - Virtual keyboard with vowels, consonants, marks, numbers
- **Transliteration** - ITRANS to Telugu conversion (e.g., "namaste" → నమస్తే)
- **Answer Display** - Raw model output and morphology-refined answer
- **TTS Playback** - Listen to answers in Telugu
- **Training Charts** - Visualize model training loss curves
- **Responsive Design** - Works on desktop and mobile

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Recharts** - Training charts
- **Lucide React** - Icons

## Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

## API Connection

The frontend connects to the FastAPI backend at `http://localhost:8000`.

Make sure the backend is running:
```bash
cd backend
uvicorn main:app --reload
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ModelSelector.tsx    # Model dropdown
│   │   ├── InputPanel.tsx       # Text input fields
│   │   ├── VoiceInput.tsx       # Voice recording
│   │   ├── TeluguKeyboard.tsx   # Virtual keyboard
│   │   ├── AnswerDisplay.tsx    # Answer with refinement
│   │   └── TrainingChart.tsx    # Loss charts
│   ├── api.ts                   # API service
│   ├── types.ts                 # TypeScript types
│   ├── App.tsx                  # Main app
│   ├── main.tsx                 # Entry point
│   └── index.css                # Tailwind styles
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```
