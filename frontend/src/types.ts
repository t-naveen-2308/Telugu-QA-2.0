// API Types for Telugu QA System

export interface ModelInfo {
  key: string;
  name: string;
  description: string;
  exact_match: number;
  f1_score: number;
  recommended: boolean;
}

export interface QARequest {
  context: string;
  question: string;
  model_key: string;
}

export interface QAResponse {
  answer: string;
  confidence: number;
  start: number;
  end: number;
  original_answer: string;
  original_confidence: number;
  refinement_applied: boolean;
  removed_suffixes: string[];
  confidence_improvement: number;
  model_key: string;
  is_relevant: boolean;
}

export interface TransliterateResponse {
  original: string;
  options: string[];
}

export interface TranscriptionResponse {
  text: string;
  language: string;
  duration?: number;
}

export interface Example {
  context: string;
  question: string;
}

export interface Examples {
  [key: string]: Example;
}

export interface ModelTrainingLoss {
  steps: number[];
  training_loss: number[];
  validation_loss: number[];
  best_step: number;
  best_val_loss: number;
}

export interface TrainingData {
  muril: ModelTrainingLoss;
  xlmr: ModelTrainingLoss;
  indicbert: ModelTrainingLoss;
  mbert: ModelTrainingLoss;
  muril_domain?: ModelTrainingLoss;
  mbert_domain?: ModelTrainingLoss;
}

// Telugu keyboard types
export interface TeluguChar {
  char: string;
  name: string;
}

export const TELUGU_VOWELS: TeluguChar[] = [
  { char: 'అ', name: 'a' }, { char: 'ఆ', name: 'aa' }, { char: 'ఇ', name: 'i' },
  { char: 'ఈ', name: 'ee' }, { char: 'ఉ', name: 'u' }, { char: 'ఊ', name: 'oo' },
  { char: 'ఋ', name: 'ru' }, { char: 'ఎ', name: 'e' }, { char: 'ఏ', name: 'ae' },
  { char: 'ఐ', name: 'ai' }, { char: 'ఒ', name: 'o' }, { char: 'ఓ', name: 'oe' },
  { char: 'ఔ', name: 'au' }, { char: 'అం', name: 'am' }, { char: 'అః', name: 'ah' }
];

export const TELUGU_CONSONANTS: TeluguChar[] = [
  // క వర్గం (ka varga)
  { char: 'క', name: 'ka' }, { char: 'ఖ', name: 'kha' }, { char: 'గ', name: 'ga' },
  { char: 'ఘ', name: 'gha' }, { char: 'ఙ', name: 'nga' },
  // చ వర్గం (cha varga)
  { char: 'చ', name: 'cha' }, { char: 'ఛ', name: 'chha' }, { char: 'జ', name: 'ja' },
  { char: 'ఝ', name: 'jha' }, { char: 'ఞ', name: 'nya' },
  // ట వర్గం (ta varga)
  { char: 'ట', name: 'ta' }, { char: 'ఠ', name: 'tha' }, { char: 'డ', name: 'da' },
  { char: 'ఢ', name: 'dha' }, { char: 'ణ', name: 'na' },
  // త వర్గం (tha varga)
  { char: 'త', name: 'tha' }, { char: 'థ', name: 'thha' }, { char: 'ద', name: 'dha' },
  { char: 'ధ', name: 'dhha' }, { char: 'న', name: 'na' },
  // ప వర్గం (pa varga)
  { char: 'ప', name: 'pa' }, { char: 'ఫ', name: 'pha' }, { char: 'బ', name: 'ba' },
  { char: 'భ', name: 'bha' }, { char: 'మ', name: 'ma' },
  // య వర్గం (ya varga)
  { char: 'య', name: 'ya' }, { char: 'ర', name: 'ra' }, { char: 'ల', name: 'la' },
  { char: 'వ', name: 'va' }, { char: 'ళ', name: 'La' },
  // శ వర్గం (sha varga)
  { char: 'శ', name: 'sha' }, { char: 'ష', name: 'Sha' }, { char: 'స', name: 'sa' },
  { char: 'హ', name: 'ha' }, { char: 'క్ష', name: 'ksha' }
];

export const TELUGU_VOWEL_MARKS: TeluguChar[] = [
  { char: 'ా', name: 'aa' }, { char: 'ి', name: 'i' }, { char: 'ీ', name: 'ee' },
  { char: 'ు', name: 'u' }, { char: 'ూ', name: 'oo' }, { char: 'ృ', name: 'ru' },
  { char: 'ె', name: 'e' }, { char: 'ే', name: 'ae' }, { char: 'ై', name: 'ai' },
  { char: 'ొ', name: 'o' }, { char: 'ో', name: 'oe' }, { char: 'ౌ', name: 'au' },
  { char: 'ం', name: 'am' }, { char: 'ః', name: 'ah' }, { char: '్', name: 'halant' }
];

export const TELUGU_SPECIAL: TeluguChar[] = [
  { char: ' ', name: 'space' }, { char: '।', name: '.' }, { char: '?', name: '?' },
  { char: '!', name: '!' }, { char: ',', name: ',' }, { char: ':', name: ':' },
  { char: ';', name: ';' }, { char: '"', name: '"' }, { char: "'", name: "'" },
  { char: '॥', name: '||' }
];
