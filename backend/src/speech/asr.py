"""
Telugu Speech Module - ASR (Automatic Speech Recognition)

Supports multiple providers with fallback:
1. Sarvam AI Saaras v3 (primary - best for Telugu)
2. OpenAI Whisper (fallback)
3. Google Speech-to-Text (final fallback)

Provider is configured via ASR_PROVIDER env variable.
"""

import os
from pathlib import Path
from typing import Optional, Union
import tempfile

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import unicodedata


class TeluguASR:
    """
    Telugu Automatic Speech Recognition with multi-provider support.
    
    Providers (configured via ASR_PROVIDER env var):
    - sarvam: Sarvam AI Saaras v3 (best Telugu quality)
    - openai: OpenAI Whisper (good multilingual)
    - google: Google Speech-to-Text / SpeechRecognition (free fallback)
    
    Automatic fallback chain: sarvam -> openai -> google
    """
    
    PROVIDERS = ['sarvam', 'openai', 'google']
    
    def __init__(
        self,
        provider: Optional[str] = None,
        sarvam_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize ASR engine.
        
        Args:
            provider: ASR provider ('sarvam', 'openai', 'google'). Uses ASR_PROVIDER env var if not set.
            sarvam_api_key: Sarvam AI API key (uses SARVAM_API_KEY env var if not provided)
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.provider = (provider or os.getenv("ASR_PROVIDER", "sarvam")).lower()
        self.sarvam_api_key = sarvam_api_key or os.getenv("SARVAM_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        self._sarvam_client = None
        self._openai_client = None
        self._google_recognizer = None
        
        self._init_provider(self.provider)
    
    def _init_provider(self, provider: str):
        """Initialize the specified provider."""
        if provider == "sarvam":
            self._init_sarvam()
        elif provider == "openai":
            self._init_openai()
        elif provider == "google":
            self._init_google()
        else:
            raise ValueError(f"Unknown ASR provider: {provider}. Use: {self.PROVIDERS}")
    
    def _init_sarvam(self):
        """Initialize Sarvam AI ASR."""
        if not self.sarvam_api_key:
            raise ValueError("Sarvam API key not found. Set SARVAM_API_KEY environment variable.")
        
        try:
            from sarvamai import SarvamAI
            self._sarvam_client = SarvamAI(api_subscription_key=self.sarvam_api_key)
            self._sarvam_use_sdk = True
        except ImportError:
            import requests
            self._sarvam_use_sdk = False
            self._sarvam_session = requests.Session()
            self._sarvam_session.headers.update({"api-subscription-key": self.sarvam_api_key})
        
        print("✓ Telugu ASR initialized (Sarvam AI Saaras v3)")
    
    def _init_openai(self):
        """Initialize OpenAI Whisper ASR."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.openai_api_key)
            print("✓ Telugu ASR initialized (OpenAI Whisper)")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _init_google(self):
        """Initialize Google Speech Recognition (free, offline-capable)."""
        try:
            import speech_recognition as sr
            self._google_recognizer = sr.Recognizer()
            print("✓ Telugu ASR initialized (Google Speech Recognition - Free)")
        except ImportError:
            raise ImportError("SpeechRecognition package not installed. Run: pip install SpeechRecognition")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        prompt: Optional[str] = None,
        language_code: str = "te-IN"
    ) -> dict:
        """
        Transcribe Telugu audio to text with automatic fallback.
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            prompt: Optional prompt (for Whisper context)
            language_code: BCP-47 language code (default: te-IN for Telugu)
            
        Returns:
            Dictionary with 'text', 'provider', and metadata
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Try primary provider first, then fallback
        providers_to_try = [self.provider] + [p for p in self.PROVIDERS if p != self.provider]
        last_error = None
        
        for provider in providers_to_try:
            try:
                if provider == "sarvam" and (self._sarvam_client or hasattr(self, '_sarvam_session')):
                    return self._transcribe_sarvam(audio_path, language_code)
                elif provider == "openai" and self._openai_client:
                    return self._transcribe_openai(audio_path, prompt, language_code)
                elif provider == "google":
                    if not self._google_recognizer:
                        self._init_google()
                    return self._transcribe_google(audio_path, language_code)
            except Exception as e:
                last_error = e
                print(f"⚠ {provider} ASR failed: {e}, trying next...")
                # Initialize next provider if not already
                try:
                    next_idx = providers_to_try.index(provider) + 1
                    if next_idx < len(providers_to_try):
                        self._init_provider(providers_to_try[next_idx])
                except:
                    pass
                continue
        
        raise RuntimeError(f"All ASR providers failed. Last error: {last_error}")
    
    def _transcribe_sarvam(self, audio_path: Path, language_code: str) -> dict:
        """Transcribe using Sarvam AI."""
        if self._sarvam_use_sdk:
            with open(audio_path, "rb") as audio_file:
                response = self._sarvam_client.speech_to_text.transcribe(
                    file=audio_file,
                    model="saaras:v3",
                    mode="transcribe",
                    language_code=language_code
                )
            text = response.transcript if hasattr(response, 'transcript') else str(response)
        else:
            url = "https://api.sarvam.ai/speech-to-text"
            with open(audio_path, "rb") as audio_file:
                files = {"file": (audio_path.name, audio_file)}
                data = {"model": "saaras:v3", "mode": "transcribe", "language_code": language_code}
                response = self._sarvam_session.post(url, files=files, data=data)
                response.raise_for_status()
                text = response.json().get("transcript", "")
        
        text = unicodedata.normalize("NFC", text)
        return {"text": text, "provider": "sarvam", "language": language_code, "confidence": 1.0}
    
    def _transcribe_openai(self, audio_path: Path, prompt: Optional[str], language_code: str) -> dict:
        """Transcribe using OpenAI Whisper."""
        with open(audio_path, "rb") as audio_file:
            response = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="te" if language_code.startswith("te") else language_code[:2],
                prompt=prompt
            )
        
        text = unicodedata.normalize("NFC", response.text)
        return {"text": text, "provider": "openai", "language": language_code, "confidence": 1.0}
    
    def _transcribe_google(self, audio_path: Path, language_code: str) -> dict:
        """Transcribe using Google Speech Recognition (free)."""
        import speech_recognition as sr
        
        # Convert to WAV if needed
        wav_path = audio_path
        if audio_path.suffix.lower() not in ['.wav', '.wave']:
            wav_path = Path(tempfile.mktemp(suffix=".wav"))
            self._convert_to_wav(audio_path, wav_path)
        
        with sr.AudioFile(str(wav_path)) as source:
            audio = self._google_recognizer.record(source)
        
        # Map language code
        google_lang = "te-IN" if language_code.startswith("te") else language_code
        text = self._google_recognizer.recognize_google(audio, language=google_lang)
        text = unicodedata.normalize("NFC", text)
        
        return {"text": text, "provider": "google", "language": language_code, "confidence": 0.9}
    
    def _convert_to_wav(self, input_path: Path, output_path: Path):
        """Convert audio to WAV format."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(input_path))
            audio.export(str(output_path), format="wav")
        except ImportError:
            raise ImportError("pydub required for audio conversion. Run: pip install pydub")
    
    def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> dict:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Raw audio bytes
            filename: Original filename (for format detection)
        """
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            return self.transcribe(temp_path)
        finally:
            os.unlink(temp_path)
    
    def set_provider(self, provider: str):
        """Switch to a different provider at runtime."""
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Use: {self.PROVIDERS}")
        self.provider = provider
        self._init_provider(provider)


def load_asr(provider: Optional[str] = None) -> TeluguASR:
    """Convenience function to load ASR engine."""
    return TeluguASR(provider=provider)


# Quick test
if __name__ == "__main__":
    print("Telugu ASR Module (Multi-provider)")
    print("=" * 40)
    print(f"Available providers: {TeluguASR.PROVIDERS}")
    print(f"Current: ASR_PROVIDER={os.getenv('ASR_PROVIDER', 'sarvam')}")
    
    try:
        asr = load_asr()
        print("✓ ASR engine ready")
    except Exception as e:
        print(f"Error: {e}")
