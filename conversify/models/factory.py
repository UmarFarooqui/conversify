"""Provider-switching factories for STT and TTS.

The active provider is chosen from config (`stt.provider` / `tts.provider`),
which lets the old Whisper/Kokoro implementations stay in place for A/B testing
alongside the new Moonshine/Piper adapters.
"""
import logging
from typing import Any, Dict

from livekit.agents import stt as lk_stt
from livekit.agents import tts as lk_tts

logger = logging.getLogger(__name__)

# Defaults preserve prior behavior if a `provider` key is ever missing.
DEFAULT_STT_PROVIDER = "moonshine"
DEFAULT_TTS_PROVIDER = "piper"


def build_stt(config: Dict[str, Any]) -> lk_stt.STT:
    """Construct the STT provider selected in config['stt']['provider']."""
    provider = config['stt'].get('provider', DEFAULT_STT_PROVIDER)
    logger.info(f"Building STT provider: {provider}")

    if provider == "moonshine":
        from .stt import MoonshineSTT
        return MoonshineSTT(config=config)
    if provider == "whisper":
        from .stt import WhisperSTT
        return WhisperSTT(config=config)

    raise ValueError(
        f"Unknown stt.provider '{provider}'. Expected 'moonshine' or 'whisper'."
    )


def build_tts(config: Dict[str, Any]) -> lk_tts.TTS:
    """Construct the TTS provider selected in config['tts']['provider']."""
    provider = config['tts'].get('provider', DEFAULT_TTS_PROVIDER)
    logger.info(f"Building TTS provider: {provider}")

    if provider == "piper":
        from .tts import PiperTTS
        return PiperTTS(config=config)
    if provider == "kokoro":
        from .tts import KokoroTTS
        return KokoroTTS(config=config)

    raise ValueError(
        f"Unknown tts.provider '{provider}'. Expected 'piper' or 'kokoro'."
    )
