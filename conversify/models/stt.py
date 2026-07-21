import asyncio
import dataclasses
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, combine_frames

from .utils import MoonshineModels, WhisperModels, find_time

logger = logging.getLogger(__name__)

# Moonshine (and Whisper) operate on 16 kHz mono audio.
MOONSHINE_SAMPLE_RATE = 16000


def _resample_pcm(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio using scipy's polyphase filter.

    Chosen over librosa.resample to avoid a numba JIT compile on first use.
    """
    if src_rate == dst_rate:
        return samples
    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(int(src_rate), int(dst_rate))
    up = int(dst_rate) // g
    down = int(src_rate) // g
    return resample_poly(samples, up, down).astype(np.float32)

@dataclass
class WhisperOptions:
    """Configuration options for WhisperSTT."""
    language: str
    model: WhisperModels | str
    device: str | None
    compute_type: str | None
    model_cache_directory: str | None
    warmup_audio: str | None


class WhisperSTT(stt.STT):
    """STT implementation using Whisper model."""
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """Initialize the WhisperSTT instance.
        
        Args:
            config: Configuration dictionary (from config.yaml)
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
               
        stt_config = config['stt']['whisper']
        
        language = stt_config['language']
        model = stt_config['model']
        device = stt_config['device']
        compute_type = stt_config['compute_type']
        model_cache_directory = stt_config['model_cache_directory']
        warmup_audio = stt_config['warmup_audio']

        self._opts = WhisperOptions(
            language=language,
            model=model,
            device=device,
            compute_type=compute_type,
            model_cache_directory=model_cache_directory,
            warmup_audio=warmup_audio
        )
        
        self._model = None
        self._initialize_model()
        
        # Warmup the model with a sample audio if available
        if warmup_audio and os.path.exists(warmup_audio):
            self._warmup(warmup_audio)

    def _initialize_model(self):
        """Initialize the Whisper model."""
        # Imported lazily so that merely importing this module (e.g. when the
        # active STT provider is Moonshine) does not pull in faster_whisper and
        # its huggingface_hub side effects.
        from faster_whisper import WhisperModel

        device = self._opts.device
        compute_type = self._opts.compute_type

        logger.info(f"Using device: {device}, with compute: {compute_type}")
        
        # Ensure cache directories exist
        model_cache_dir = self._opts.model_cache_directory
        
        if model_cache_dir:
            os.makedirs(model_cache_dir, exist_ok=True)
            logger.info(f"Using model cache directory: {model_cache_dir}")
        
        self._model = WhisperModel(
            model_size_or_path=str(self._opts.model),
            device=device,
            compute_type=compute_type,
            download_root=model_cache_dir
        )
        logger.info("Whisper model loaded successfully")

    def _warmup(self, warmup_audio_path: str) -> None:
        """Performs a warmup transcription.
        
        Args:
            warmup_audio_path: Path to audio file for warmup
        """
        logger.info(f"Starting STT engine warmup using {warmup_audio_path}...")
        try:
            with find_time('STT_warmup'):
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                segments, info = self._model.transcribe(warmup_audio_data, 
                                                        language=self._opts.language, 
                                                        beam_size=1)
                model_warmup_transcription = " ".join(segment.text for segment in segments)
            logger.info(f"STT engine warmed up. Text: {model_warmup_transcription}")
        except Exception as e:
            logger.error(f"Failed to warm up STT engine: {e}")

    def update_options(
        self,
        *,
        model: Optional[WhisperModels | str] = None,
        language: Optional[str] = None,
        model_cache_directory: Optional[str] = None,
    ) -> None:
        """Update STT options.
        
        Args:
            model: Whisper model to use
            language: Language to detect
            model_cache_directory: Directory to store downloaded models
        """
        reinitialize = False
        
        if model:
            self._opts.model = model
            reinitialize = True
            
        if model_cache_directory:
            self._opts.model_cache_directory = model_cache_directory
            reinitialize = True
            
        if language:
            self._opts.language = language
            
        if reinitialize:
            self._initialize_model()

    def _sanitize_options(self, *, language: Optional[str] = None) -> WhisperOptions:
        """Create a copy of options with optional overrides.
        
        Args:
            language: Language override
            
        Returns:
            Copy of options with overrides applied
        """
        options = dataclasses.replace(self._opts)
        if language:
            options.language = language
        return options

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: Optional[str],
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Implement speech recognition.
        
        Args:
            buffer: Audio buffer
            language: Language to detect
            conn_options: Connection options
            
        Returns:
            Speech recognition event
        """
        try:
            logger.info(f"Received audio, transcribing to text")
            options = self._sanitize_options(language=language)
            audio_data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            
            # Convert WAV to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            with find_time('STT_inference'):
                segments, info = self._model.transcribe(
                    audio_array,
                    language=options.language,
                    beam_size=1,
                    best_of=1,
                    condition_on_previous_text=True,
                    vad_filter=False,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )

            segments_list = list(segments)
            full_text = " ".join(segment.text.strip() for segment in segments_list)

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=full_text or "",
                        language=options.language,
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Error in speech recognition: {e}", exc_info=True)
            raise APIConnectionError() from e


@dataclass
class MoonshineOptions:
    """Configuration options for MoonshineSTT."""
    language: str
    model: MoonshineModels | str
    device: str | None
    model_cache_directory: str | None


class MoonshineSTT(stt.STT):
    """STT implementation using the Moonshine ONNX model.

    Moonshine is a non-streaming, CPU-friendly recognizer. As with WhisperSTT we
    declare ``streaming=False``/``interim_results=False`` so LiveKit wraps this in
    its VAD-based ``StreamAdapter`` (the Silero VAD provided in prewarm), keeping
    the existing endpointing/turn-detection wiring intact.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """Initialize the MoonshineSTT instance.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        stt_config = config['stt']['moonshine']

        language = stt_config['language']
        model = stt_config['model']
        device = stt_config.get('device')
        model_cache_directory = stt_config.get('model_cache_directory')

        self._opts = MoonshineOptions(
            language=language,
            model=model,
            device=device,
            model_cache_directory=model_cache_directory,
        )

        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Load the Moonshine ONNX model once, at construction time."""
        # moonshine_onnx downloads its weights via huggingface_hub and does NOT
        # accept a cache-directory argument. The only way to honor
        # ``model_cache_directory`` is to point the HF cache env vars at it
        # *before the first import of moonshine_onnx (or huggingface_hub)*.
        # We set them here and import moonshine_onnx lazily right after; if
        # huggingface_hub was already imported earlier in the process (e.g. by
        # another provider) these will be ignored, so for a guaranteed cache
        # location export HF_HOME / HF_HUB_CACHE in the environment instead.
        cache_dir = self._opts.model_cache_directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            os.environ.setdefault("HF_HOME", cache_dir)
            os.environ.setdefault("HF_HUB_CACHE", cache_dir)
            logger.info(f"Using model cache directory (HF cache): {cache_dir}")

        import moonshine_onnx

        self._moonshine_onnx = moonshine_onnx
        logger.info(f"Loading Moonshine model '{self._opts.model}' (device={self._opts.device})")
        # onnxruntime selects the CPU provider by default; the `device` option is
        # informational here since moonshine_onnx does not expose provider choice.
        self._model = moonshine_onnx.MoonshineOnnxModel(model_name=str(self._opts.model))
        logger.info("Moonshine model loaded successfully")
        self._warmup()

    def _warmup(self) -> None:
        """Run one throwaway transcription so the first real turn isn't slow.

        Warms the ONNX encoder/decoder graphs (and, via _recognize_impl's path on
        the first live call, keeps latency predictable).
        """
        try:
            with find_time('STT_warmup'):
                # 0.5 s of silence at 16 kHz satisfies Moonshine's 0.1s..64s bound.
                silence = np.zeros(MOONSHINE_SAMPLE_RATE // 2, dtype=np.float32)
                self._transcribe(silence)
            logger.info("Moonshine warmup complete")
        except Exception as e:
            logger.warning(f"Moonshine warmup skipped: {e}")

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run the synchronous, CPU-bound Moonshine transcription.

        Args:
            audio: 16 kHz mono float32 samples in [-1, 1]

        Returns:
            The first transcription string, or "" if none was produced.
        """
        # transcribe() accepts a numpy array directly (no temp WAV needed) and a
        # preloaded model object, and returns a list of strings.
        results = self._moonshine_onnx.transcribe(audio, self._model)
        return results[0] if results else ""

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Implement speech recognition.

        Args:
            buffer: Audio buffer delivered by LiveKit (int16 at the room's rate)
            language: Language override (Moonshine is English-only; used for tagging)
            conn_options: Connection options

        Returns:
            Speech recognition event with a single FINAL_TRANSCRIPT alternative
        """
        try:
            lang = language if isinstance(language, str) else self._opts.language

            # Merge all frames of the utterance into one, then read the real
            # sample rate from the frame rather than assuming it.
            frame = combine_frames(buffer)
            src_rate = frame.sample_rate

            # int16 -> float32 normalized to [-1, 1]
            samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Downmix to mono if the frame carries multiple channels.
            if frame.num_channels > 1:
                samples = samples.reshape(-1, frame.num_channels).mean(axis=1)

            # Resample to 16 kHz only if needed. We use scipy's polyphase
            # resampler rather than librosa.resample: librosa's default path
            # pulls in numba and JIT-compiles on the first call, which stalls the
            # very first recognition for many seconds (and floods DEBUG logs).
            if src_rate != MOONSHINE_SAMPLE_RATE:
                samples = _resample_pcm(samples, src_rate, MOONSHINE_SAMPLE_RATE)

            # Moonshine only accepts 0.1s..64s segments; skip anything shorter.
            duration = samples.shape[0] / MOONSHINE_SAMPLE_RATE
            if duration < 0.1:
                logger.debug(f"Audio too short for Moonshine ({duration:.3f}s), returning empty transcript")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language=lang)],
                )

            logger.info(f"Received audio ({duration:.2f}s @ {src_rate}Hz), transcribing (Moonshine)")
            # transcribe() is synchronous and CPU-bound: run it off the event loop.
            loop = asyncio.get_event_loop()
            with find_time('STT_inference'):
                full_text = await loop.run_in_executor(None, self._transcribe, samples)
            logger.info(f"Moonshine transcript: {full_text!r}")

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        text=full_text or "",
                        language=lang,
                    )
                ],
            )

        except Exception as e:
            logger.error(f"Error in speech recognition: {e}", exc_info=True)
            raise APIConnectionError() from e 