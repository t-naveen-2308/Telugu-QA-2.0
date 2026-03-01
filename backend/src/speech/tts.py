"""
Telugu Speech Module - TTS (Text-to-Speech)

Supports multiple providers with fallback:
1. Sarvam AI Bulbul v3 (primary - best Telugu quality)
2. OpenAI TTS (fallback)
3. Google gTTS (final fallback - free)

Provider is configured via TTS_PROVIDER env variable.
"""

import os
from pathlib import Path
from typing import Optional
import tempfile
import io
import base64

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TeluguTTS:
    """
    Telugu Text-to-Speech with multi-provider support.
    
    Providers (configured via TTS_PROVIDER env var):
    - sarvam: Sarvam AI Bulbul v3 (best Telugu quality, 30+ voices)
    - openai: OpenAI TTS (good quality, limited Telugu)
    - google: Google gTTS (free, decent quality)
    
    Automatic fallback chain: sarvam -> openai -> google
    """
    
    PROVIDERS = ['sarvam', 'openai', 'google']
    
    # Sarvam AI speakers (Bulbul v3)
    SARVAM_SPEAKERS = [
        "anushka", "abhilash", "manisha", "vidya", "arya", "karun", 
        "hitesh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", 
        "rohan", "simran", "kavya", "amit", "dev", "ishita", "shreya",
        "kavitha", "meera"
    ]
    
    # OpenAI TTS voices
    OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    def __init__(
        self,
        provider: Optional[str] = None,
        sarvam_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        speaker: str = "kavitha"
    ):
        """
        Initialize TTS engine.
        
        Args:
            provider: TTS provider ('sarvam', 'openai', 'google'). Uses TTS_PROVIDER env var if not set.
            sarvam_api_key: Sarvam AI API key
            openai_api_key: OpenAI API key  
            speaker: Voice to use (provider-specific)
        """
        self.provider = (provider or os.getenv("TTS_PROVIDER", "sarvam")).lower()
        self.sarvam_api_key = sarvam_api_key or os.getenv("SARVAM_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.speaker = speaker
        self.language = "te-IN"
        
        self._sarvam_client = None
        self._openai_client = None
        self._gtts_available = False
        
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
            raise ValueError(f"Unknown TTS provider: {provider}. Use: {self.PROVIDERS}")
    
    def _init_sarvam(self):
        """Initialize Sarvam AI TTS."""
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
            self._sarvam_session.headers.update({
                "api-subscription-key": self.sarvam_api_key,
                "Content-Type": "application/json"
            })
        
        print(f"✓ Telugu TTS initialized (Sarvam AI Bulbul v3 - {self.speaker})")
    
    def _init_openai(self):
        """Initialize OpenAI TTS."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=self.openai_api_key)
            # Map speaker to OpenAI voice
            if self.speaker not in self.OPENAI_VOICES:
                self.speaker = "nova"  # Default female voice
            print(f"✓ Telugu TTS initialized (OpenAI TTS - {self.speaker})")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _init_google(self):
        """Initialize Google gTTS (free)."""
        try:
            from gtts import gTTS
            self._gtts_available = True
            print("✓ Telugu TTS initialized (Google gTTS - Free)")
        except ImportError:
            raise ImportError("gTTS package not installed. Run: pip install gTTS")
    
    def speak(
        self,
        text: str,
        output_path: Optional[str] = None,
        slow: bool = False
    ) -> str:
        """
        Convert Telugu text to speech and save to file.
        
        Args:
            text: Telugu text to speak
            output_path: Where to save the audio (auto-generated if None)
            slow: Speak slowly
            
        Returns:
            Path to the generated audio file
        """
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp3")
        
        audio_bytes = self.speak_bytes(text, slow=slow)
        
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        
        return output_path
    
    def speak_bytes(
        self,
        text: str,
        slow: bool = False,
        pace: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        Convert Telugu text to speech with automatic fallback.
        
        Args:
            text: Telugu text to speak
            slow: Speak slowly (sets pace=0.8)
            pace: Speech pace (0.5 to 2.0, default 1.0)
            
        Returns:
            Audio bytes (WAV/MP3 format depending on provider)
        """
        if slow:
            pace = 0.8
        
        # Try primary provider first, then fallback
        providers_to_try = [self.provider] + [p for p in self.PROVIDERS if p != self.provider]
        last_error = None
        
        for provider in providers_to_try:
            try:
                if provider == "sarvam" and (self._sarvam_client or hasattr(self, '_sarvam_session')):
                    return self._speak_sarvam(text, pace)
                elif provider == "openai" and self._openai_client:
                    return self._speak_openai(text, pace)
                elif provider == "google":
                    if not self._gtts_available:
                        self._init_google()
                    return self._speak_google(text, slow)
            except Exception as e:
                last_error = e
                print(f"⚠ {provider} TTS failed: {e}, trying next...")
                try:
                    next_idx = providers_to_try.index(provider) + 1
                    if next_idx < len(providers_to_try):
                        self._init_provider(providers_to_try[next_idx])
                except:
                    pass
                continue
        
        raise RuntimeError(f"All TTS providers failed. Last error: {last_error}")
    
    def _speak_sarvam(self, text: str, pace: float) -> bytes:
        """Generate speech using Sarvam AI."""
        if self._sarvam_use_sdk:
            response = self._sarvam_client.text_to_speech.convert(
                target_language_code=self.language,
                text=text,
                model="bulbul:v3",
                speaker=self.speaker,
                pace=pace
            )
            if hasattr(response, 'audios') and response.audios:
                return base64.b64decode(response.audios[0])
            elif hasattr(response, 'audio'):
                return base64.b64decode(response.audio)
            else:
                return base64.b64decode(str(response))
        else:
            url = "https://api.sarvam.ai/text-to-speech"
            payload = {
                "target_language_code": self.language,
                "text": text,
                "model": "bulbul:v3",
                "speaker": self.speaker,
                "pace": pace
            }
            response = self._sarvam_session.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "audios" in result and result["audios"]:
                return base64.b64decode(result["audios"][0])
            elif "audio" in result:
                return base64.b64decode(result["audio"])
            else:
                raise ValueError("No audio in response")
    
    def _speak_openai(self, text: str, pace: float) -> bytes:
        """Generate speech using OpenAI TTS."""
        # OpenAI doesn't support pace directly, but we can adjust speed
        speed = pace
        
        response = self._openai_client.audio.speech.create(
            model="tts-1",
            voice=self.speaker if self.speaker in self.OPENAI_VOICES else "nova",
            input=text,
            speed=speed
        )
        
        return response.content
    
    def _speak_google(self, text: str, slow: bool) -> bytes:
        """Generate speech using Google gTTS (free)."""
        from gtts import gTTS
        
        tts = gTTS(text=text, lang='te', slow=slow)
        
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        return mp3_fp.read()
    
    def set_provider(self, provider: str):
        """Switch to a different provider at runtime."""
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Use: {self.PROVIDERS}")
        self.provider = provider
        self._init_provider(provider)


def load_tts(provider: Optional[str] = None) -> TeluguTTS:
    """Convenience function to load TTS engine."""
    return TeluguTTS(provider=provider)


# Quick test
if __name__ == "__main__":
    print("Telugu TTS Module (Multi-provider)")
    print("=" * 40)
    print(f"Available providers: {TeluguTTS.PROVIDERS}")
    print(f"Current: TTS_PROVIDER={os.getenv('TTS_PROVIDER', 'sarvam')}")
    
    try:
        tts = load_tts()
        print("✓ TTS engine ready")
        
        # Quick test
        print("\nTesting...")
        audio = tts.speak_bytes("హలో")
        print(f"✓ Generated {len(audio)} bytes of audio")
    except Exception as e:
        print(f"Error: {e}")
