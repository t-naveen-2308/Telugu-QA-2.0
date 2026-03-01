// API Service for Telugu QA System

import type {
  ModelInfo,
  QARequest,
  QAResponse,
  TransliterateResponse,
  TranscriptionResponse,
  Examples,
  TrainingData
} from './types';

const API_BASE = 'http://localhost:8000';

class ApiService {
  private async fetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // List available models
  async getModels(): Promise<ModelInfo[]> {
    return this.fetch<ModelInfo[]>('/models');
  }

  // Get QA prediction
  async predict(request: QARequest): Promise<QAResponse> {
    return this.fetch<QAResponse>('/qa/predict', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Transliterate ITRANS to Telugu
  async transliterate(text: string): Promise<TransliterateResponse> {
    return this.fetch<TransliterateResponse>('/transliterate', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  }

  // Get TTS audio
  async getTTS(text: string, slow: boolean = false): Promise<Blob> {
    const response = await fetch(`${API_BASE}/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, slow }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'TTS error' }));
      throw new Error(error.detail);
    }

    return response.blob();
  }

  // Transcribe audio
  async transcribe(audioBlob: Blob, filename: string = 'audio.wav'): Promise<TranscriptionResponse> {
    const formData = new FormData();
    formData.append('file', audioBlob, filename);

    const response = await fetch(`${API_BASE}/asr/transcribe`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Transcription error' }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  // Get example contexts
  async getExamples(): Promise<Examples> {
    return this.fetch<Examples>('/examples');
  }

  // Get training data for charts
  async getTrainingData(): Promise<TrainingData> {
    return this.fetch<TrainingData>('/training-data');
  }

  // Health check
  async healthCheck(): Promise<{ status: string; models_loaded: string[] }> {
    return this.fetch('/health');
  }
}

export const api = new ApiService();
