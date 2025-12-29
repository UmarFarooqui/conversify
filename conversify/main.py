import logging
import asyncio
import os
from dotenv import load_dotenv
from typing import Dict, Any
from openai import AsyncClient
import functools 

from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    JobExecutorType,
    RoomInputOptions,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
    metrics
)
from livekit.plugins import silero
from livekit.agents.types import NOT_GIVEN
from livekit.plugins import noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

try:
    from livekit.plugins import bithuman
    from PIL import Image
    BITHUMAN_AVAILABLE = True
except ImportError:
    BITHUMAN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.debug("bithuman plugin not available. Install with: pip install 'livekit-agents[bithuman,images]'")

from .models.tts import KokoroTTS
from .models.stt import WhisperSTT
from .models.llm import OpenaiLLM
from .core.vision import video_processing_loop
from .core.agent import ConversifyAgent
from .utils.logger import setup_logging
from .utils.config import ConfigManager
from .core.callbacks import metrics_callback, shutdown_callback

logger = logging.getLogger(__name__)


def prewarm(proc: JobProcess, config: Dict[str, Any]):
    """Prewarms resources needed by the agent, like the VAD model."""
    logger.info("Prewarming VAD...")    
    vad_config = config['vad']
    
    proc.userdata["vad"] = silero.VAD.load(
                                min_speech_duration=vad_config['min_speech_duration'],
                                min_silence_duration=vad_config['min_silence_duration'],
                                prefix_padding_duration=vad_config['prefix_padding_duration'],
                                max_buffered_speech=vad_config['max_buffered_speech'],
                                activation_threshold=vad_config['activation_threshold'],
                                force_cpu=vad_config['force_cpu'],
                                sample_rate=vad_config['sample_rate']
                        )
    logger.info("VAD prewarmed successfully.")


async def entrypoint(ctx: JobContext, config: Dict[str, Any]):
    """The main entrypoint for the agent job."""
    # Setup initial logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "job_id": ctx.job.id,
    }
    logger.info(f"Agent entrypoint started. Context: {ctx.log_context_fields}")
    
    await ctx.connect()
    logger.info("Successfully connected to room.")

    # Create shared state dictionary for inter-task communication
    shared_state: Dict[str, Any] = {}

    # Initialize LLM Client here using config
    llm_config = config['llm']
    try:
        llm_client = AsyncClient(api_key=llm_config['api_key'], base_url=llm_config['base_url'])
        logger.info(f"Initialized LLM Client at {llm_config['base_url']}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Client: {e}")
        raise

    # Check if VAD was prewarmed successfully
    vad = ctx.proc.userdata.get("vad")
    if not vad:
        logger.error("VAD not found in process userdata. Exiting.")
        return

    # Setup the AgentSession with configured plugins
    session = AgentSession(
        vad=vad,
        llm=OpenaiLLM(client=llm_client, config=config), 
        stt=WhisperSTT(config=config),
        tts=KokoroTTS(config=config),
        turn_detection=MultilingualModel() if config['agent']['use_eou'] else NOT_GIVEN
    )
    logger.info("AgentSession created.")

    # Initialize bithuman avatar if configured
    avatar_session = None
    avatar_config = config.get('avatar', {})
    if avatar_config.get('use', False) and BITHUMAN_AVAILABLE:
        logger.info("Initializing bitHuman avatar...")
        bithuman_config = avatar_config['bithuman']
        
        # Get API secret from environment
        api_secret = os.getenv('BITHUMAN_API_SECRET', bithuman_config.get('api_secret', ''))
        if not api_secret or api_secret == 'BITHUMAN_API_SECRET':
            logger.warning("BITHUMAN_API_SECRET not set in environment. Avatar will not be initialized.")
        else:
            try:
                # Determine which avatar method to use
                if bithuman_config.get('avatar_id'):
                    avatar_session = bithuman.AvatarSession(
                        model=bithuman_config['model'],
                        avatar_id=bithuman_config['avatar_id']
                    )
                    logger.info(f"Avatar initialized with avatar_id: {bithuman_config['avatar_id']}")
                elif bithuman_config.get('model_path'):
                    avatar_session = bithuman.AvatarSession(
                        model=bithuman_config['model'],
                        model_path=bithuman_config['model_path']
                    )
                    logger.info(f"Avatar initialized with model_path: {bithuman_config['model_path']}")
                elif bithuman_config.get('avatar_image'):
                    avatar_image_path = bithuman_config['avatar_image']
                    avatar_session = bithuman.AvatarSession(
                        model=bithuman_config['model'],
                        avatar_image=Image.open(avatar_image_path).convert("RGB")
                    )
                    logger.info(f"Avatar initialized with image: {avatar_image_path}")
                else:
                    logger.warning("No avatar configuration method specified (avatar_id, model_path, or avatar_image).")
            except Exception as e:
                logger.error(f"Failed to initialize bitHuman avatar: {e}")
                avatar_session = None
    elif avatar_config.get('use', False) and not BITHUMAN_AVAILABLE:
        logger.warning("Avatar enabled in config but bithuman plugin not installed. Install with: pip install 'livekit-agents[bithuman,images]'")

    # Start the video processing loop if configured
    video_task: asyncio.Task | None = None
    vision_config = config['vision']
    
    if vision_config['use']:
        logger.info("Starting video processing loop...")
        video_task = asyncio.create_task(video_processing_loop(ctx, shared_state, vision_config['video_frame_interval']))

    # Setup metrics collection
    metrics_callback(session)

    # Wait for a participant to join before starting the session
    logger.info("Waiting for participant to join...")
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant '{participant.identity if participant else 'unknown'}' joined.")
    
    # setup agent instance
    agent = ConversifyAgent(
        participant_identity=participant.identity,
        shared_state=shared_state,
        config=config,
        avatar_enabled=avatar_session is not None
    )

    # Register the shutdown callback 
    ctx.add_shutdown_callback(lambda: shutdown_callback(agent, video_task, avatar_session))
    logger.info("Shutdown callback registered.")

    # Start avatar session if configured
    if avatar_session:
        logger.info("Starting bitHuman avatar session...")
        try:
            await avatar_session.start(session, room=ctx.room)
            logger.info("Avatar session started successfully.")
            # Wait for avatar participant to be ready before continuing
            logger.info("Waiting for avatar participant to join...")
            await asyncio.sleep(2.0)  # Give avatar time to initialize
        except Exception as e:
            logger.error(f"Failed to start avatar session: {e}")

    # Start the agent session
    logger.info("Starting agent session...")
    
    # When using avatar, disable audio output (avatar handles it)
    from livekit.agents.voice import room_io
    
    if avatar_session:
        logger.info("Avatar detected: starting session with audio_output=False")
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC() if config['agent']['use_background_noise_removal'] else NOT_GIVEN,
            ),
            room_output_options=RoomOutputOptions(transcription_enabled=True),
            room_options=room_io.RoomOptions(audio_output=False),
        )
    else:
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC() if config['agent']['use_background_noise_removal'] else NOT_GIVEN,
            ),
            room_output_options=RoomOutputOptions(transcription_enabled=True),
        )
    
    if config['agent']['use_background_audio']:
        background_audio = BackgroundAudioPlayer(
            # play office ambience sound looping in the background
            ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=0.8),
            # play keyboard typing sound when the agent is thinking
            thinking_sound=[
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8),
                AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING2, volume=0.7),
            ],
        )

        await background_audio.start(room=ctx.room, agent_session=session)


def main():
    """Main function that initializes and runs the applicaton."""
    # Configure basic logging BEFORE loading config
    logging.basicConfig(level="INFO", 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    initial_logger = logging.getLogger(__name__)
    initial_logger.info("Basic logging configured. Loading configuration...")

    # Load configuration 
    app_config = ConfigManager().load_config()
    initial_logger.info("Configuration loaded.")
    
    # Load env variables
    load_dotenv(app_config['agent']['env_file'])

    # Setup centralized logging using the dedicated function
    setup_logging(config=app_config, project_root=ConfigManager().project_root)

    # Now, get the properly configured logger for the main module
    logger = logging.getLogger(__name__) # Re-get logger after setup
    logger.info("Centralized logging configured. Starting LiveKit Agent application...")
    
    # Create a partial function that includes the config
    entrypoint_with_config = functools.partial(entrypoint, config=app_config)
    prewarm_with_config = functools.partial(prewarm, config=app_config)

    # Define worker options using loaded config
    worker_config = app_config['worker']
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint_with_config, 
        prewarm_fnc=prewarm_with_config,
        job_memory_warn_mb=worker_config['job_memory_warn_mb'],
        load_threshold=worker_config['load_threshold'],
        job_memory_limit_mb=worker_config['job_memory_limit_mb'],
        job_executor_type=JobExecutorType.THREAD,
    )

    # Run the CLI application
    cli.run_app(worker_options)


if __name__ == "__main__":
    main()
