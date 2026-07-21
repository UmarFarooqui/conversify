import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import openai

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .utils import TTSModels, TTSVoices, find_time

logger = logging.getLogger(__name__)

TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1

# Piper is mono; its output sample rate is read from the voice config at load time.
PIPER_CHANNELS = 1

@dataclass
class KokoroTTSOptions:
    """Configuration options for KokoroTTS."""
    model: TTSModels | str
    voice: TTSVoices | str
    speed: float


class KokoroTTS(tts.TTS):
    """TTS implementation using Kokoro API."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        client: openai.AsyncClient | None = None,
    ) -> None:
        """Initialize the KokoroTTS instance.
        
        Args:
            client: Optional pre-configured OpenAI AsyncClient
            config: Configuration dictionary (from config.yaml)
        """
        tts_config = config['tts']['kokoro']
        
        model = tts_config['model']
        voice = tts_config['voice']
        speed = tts_config['speed']
        api_key = tts_config['api_key']
        base_url = tts_config['base_url']
        
        logger.info(f"Using TTS API URL: {base_url}")

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
        )

        self._opts = KokoroTTSOptions(
            model=model,
            voice=voice,
            speed=speed,
        )

        self._client = client or openai.AsyncClient(
            max_retries=0,
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """Update TTS options.
        
        Args:
            model: TTS model to use
            voice: Voice to use
            speed: Speech speed multiplier
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(speed):
            self._opts.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "KokoroTTSStream":
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            conn_options: Connection options
            
        Returns:
            Stream of audio chunks
        """
        return KokoroTTSStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            client=self._client,
        )


class KokoroTTSStream(tts.ChunkedStream):
    """Stream implementation for KokoroTTS."""
    
    def __init__(
        self,
        *,
        tts: KokoroTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: KokoroTTSOptions,
        client: openai.AsyncClient,
    ) -> None:
        """Initialize the stream.
        
        Args:
            tts: TTS instance
            input_text: Text to synthesize
            conn_options: Connection options
            opts: TTS options
            client: OpenAI AsyncClient
        """
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = client
        self._opts = opts

    async def _run(self, output_emitter) -> None:
        """Run the TTS synthesis using the modern AudioEmitter API."""
        request_id = utils.shortuuid()

        # Initialize the emitter — this is the step the old code skipped
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=TTS_SAMPLE_RATE,
            num_channels=TTS_CHANNELS,
            mime_type="audio/pcm",
        )

        oai_stream = self._client.audio.speech.with_streaming_response.create(
            input=self.input_text,
            model=self._opts.model,
            voice=self._opts.voice,
            response_format="pcm",  # raw pcm buffers
            speed=self._opts.speed,
            timeout=httpx.Timeout(30, connect=self._conn_options.timeout),
        )

        logger.info("Kokoro -> converting text to audio")

        try:
            with find_time('TTS_inferencing'):
                async with oai_stream as stream:
                    async for data in stream.iter_bytes():
                        output_emitter.push(data)
                output_emitter.flush()
        except openai.APITimeoutError:
            raise APITimeoutError()
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status_code,
                request_id=e.request_id,
                body=e.body,
            )
        except Exception as e:
            raise APIConnectionError() from e


@dataclass
class PiperTTSOptions:
    """Configuration options for PiperTTS."""
    model: str
    model_cache_directory: Optional[str]
    speaker_id: Optional[int]
    length_scale: Optional[float]
    noise_scale: Optional[float]
    noise_w: Optional[float]


class PiperTTS(tts.TTS):
    """TTS implementation using a local Piper ONNX voice.

    The voice is loaded once at construction (not per synthesis call). Piper is a
    non-streaming, synchronous engine, so we declare ``streaming=False`` and run
    synthesis in an executor, emitting audio chunks as they become available.
    """

    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        """Initialize the PiperTTS instance.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        piper_config = config['tts']['piper']

        model = piper_config['model']
        model_cache_directory = piper_config.get('model_cache_directory')
        speaker_id = piper_config.get('speaker_id')
        length_scale = piper_config.get('length_scale')
        noise_scale = piper_config.get('noise_scale')
        noise_w = piper_config.get('noise_w')

        self._opts = PiperTTSOptions(
            model=model,
            model_cache_directory=model_cache_directory,
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
        )

        # Imported lazily so importing this module doesn't require piper/onnx.
        from piper import PiperVoice, SynthesisConfig

        # PiperVoice.load requires both the .onnx model and its .onnx.json config,
        # which sits beside the model file.
        model_path = model
        config_path = f"{model_path}.json"
        for path in (model_path, config_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Piper voice file not found: {path}")

        logger.info(f"Loading Piper voice from: {model_path}")
        self._voice = PiperVoice.load(model_path, config_path)

        # Read the actual output sample rate from the voice config rather than
        # assuming 22050. LiveKit resamples this to the room rate downstream.
        self._sample_rate = self._voice.config.sample_rate
        logger.info(f"Piper voice loaded (sample_rate={self._sample_rate} Hz, "
                    f"num_speakers={self._voice.config.num_speakers})")

        # Only pass speaker_id for genuinely multi-speaker voices; a single-speaker
        # voice must receive speaker_id=None.
        effective_speaker_id = (
            speaker_id if (self._voice.config.num_speakers > 1) else None
        )

        # length_scale is Piper's speed control: < 1 is faster, > 1 is slower
        # (the inverse of the old Kokoro `speed`, where higher was faster).
        self._syn_config = SynthesisConfig(
            speaker_id=effective_speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w,
        )

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=self._sample_rate,
            num_channels=PIPER_CHANNELS,
        )

        self._warmup()

    def _warmup(self) -> None:
        """Run one throwaway synthesis at construction.

        Piper's first synthesize() call pays a large one-time cost to initialize
        the espeak-ng phonemizer (seconds, especially when its data dir lives on a
        slow filesystem). Doing it here moves that cost off the first live turn
        (e.g. the agent greeting).
        """
        try:
            with find_time('TTS_warmup'):
                for _ in self._voice.synthesize("Ready.", self._syn_config):
                    pass
            logger.info("Piper warmup complete")
        except Exception as e:
            logger.warning(f"Piper warmup skipped: {e}")

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "PiperTTSStream":
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            conn_options: Connection options

        Returns:
            Stream of audio chunks
        """
        return PiperTTSStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )


class PiperTTSStream(tts.ChunkedStream):
    """Stream implementation for PiperTTS."""

    def __init__(
        self,
        *,
        tts: PiperTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._piper_tts = tts

    async def _run(self, output_emitter) -> None:
        """Run Piper synthesis in an executor, streaming chunks as they arrive."""
        request_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._piper_tts._sample_rate,
            num_channels=PIPER_CHANNELS,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_event_loop()
        # Bridge the blocking generator (running in a worker thread) to the async
        # emitter via a queue, so audio starts flowing before full synthesis
        # completes (Piper yields roughly one chunk per sentence).
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def _synthesize_blocking() -> None:
            try:
                for chunk in self._piper_tts._voice.synthesize(
                    self.input_text, self._piper_tts._syn_config
                ):
                    # 16-bit PCM bytes at the voice's native sample rate.
                    loop.call_soon_threadsafe(queue.put_nowait, chunk.audio_int16_bytes)
            except Exception as exc:  # forwarded to the consumer below
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        logger.info("Piper -> converting text to audio")
        fut = loop.run_in_executor(None, _synthesize_blocking)
        try:
            with find_time('TTS_inferencing'):
                while True:
                    item = await queue.get()
                    if item is sentinel:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    output_emitter.push(item)
                output_emitter.flush()
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await fut