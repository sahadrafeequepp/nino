"""WebSocket endpoint /ws/voice for real-time bidirectional voice interactions."""

import json
import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status

from nino.config import Settings, get_settings
from nino.voice.duplex import AssistantVoiceState, DuplexMode
from nino.voice.pipeline import VoicePipeline
from nino.voice.stt.whisper_adapter import FasterWhisperAdapter
from nino.voice.stt.mock_stt import MockSTTAdapter
from nino.voice.tts.piper_adapter import PiperTTSAdapter
from nino.voice.tts.mock_tts import MockTTSAdapter
from nino.voice.vad.silero_vad import SileroVAD
from nino.voice.wakeword.openwakeword_detector import OpenWakeWordDetector

logger = logging.getLogger("nino.api.voice_ws")

router = APIRouter(tags=["voice"])


def verify_ws_token(token: Optional[str], settings: Settings) -> bool:
    """Verify Bearer token constant-time for WebSocket connections."""
    if not token:
        return False
    # Strip any 'Bearer ' prefix if passed in query param
    if token.startswith("Bearer "):
        token = token[7:]
    return secrets.compare_digest(token.encode("utf-8"), settings.auth_token.encode("utf-8"))


@router.websocket("/ws/voice")
async def voice_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    settings: Settings = Depends(get_settings),
):
    """Bidirectional WebSocket endpoint for 16kHz PCM audio streaming, transcripts, and TTS playback."""
    # 1. Authenticate connection
    auth_header = websocket.headers.get("Authorization")
    extracted_token = token
    if not extracted_token and auth_header and auth_header.startswith("Bearer "):
        extracted_token = auth_header[7:]

    if not verify_ws_token(extracted_token, settings):
        logger.warning("Rejecting unauthenticated voice WebSocket connection.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info("Voice WebSocket client connected.")

    # 2. Retrieve shared services from app state
    app_state = websocket.app.state
    conv_store = getattr(app_state, "conv_store", None)
    retrieval = getattr(app_state, "retrieval", None)
    orchestrator = getattr(app_state, "tool_orchestrator", None)
    confirmation_manager = getattr(app_state, "confirmation_manager", None)
    fact_manager = getattr(app_state, "fact_manager", None)
    adapter = getattr(app_state, "adapter", None)

    # 3. Initialize voice adapters (use cached or fresh instances)
    voice_pipeline: VoicePipeline = getattr(app_state, "voice_pipeline", None)
    if voice_pipeline is None:
        try:
            stt = FasterWhisperAdapter(
                model_size_or_path=settings.stt_model,
                compute_type=settings.stt_compute_type,
                device=settings.stt_device,
            )
        except Exception as e:
            logger.warning("Could not initialize FasterWhisperAdapter: %s. Using MockSTT.", e)
            stt = MockSTTAdapter()

        try:
            tts = PiperTTSAdapter(
                voice=settings.tts_voice,
                speed=settings.tts_speed,
            )
        except Exception as e:
            logger.warning("Could not initialize PiperTTSAdapter: %s. Using MockTTS.", e)
            tts = MockTTSAdapter()

        vad = SileroVAD(threshold=settings.vad_threshold)
        wake_word = OpenWakeWordDetector(
            model_name=settings.wake_word_model,
            threshold=settings.wake_word_threshold,
        )

        duplex_mode = (
            DuplexMode.BARGE_IN
            if settings.voice_duplex_mode == "barge_in"
            else DuplexMode.HALF_DUPLEX
        )

        voice_pipeline = VoicePipeline(
            stt=stt,
            tts=tts,
            wake_word=wake_word,
            vad=vad,
            conv_store=conv_store,
            adapter=adapter,
            settings=settings,
            retrieval=retrieval,
            orchestrator=orchestrator,
            confirmation_manager=confirmation_manager,
            fact_manager=fact_manager,
            duplex_mode=duplex_mode,
        )

    # 4. Attach WebSocket event emitters
    async def send_json(data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    async def send_audio(pcm_bytes: bytes):
        try:
            await websocket.send_bytes(pcm_bytes)
        except Exception:
            pass

    # Bind pipeline callbacks to WebSocket
    voice_pipeline.on_state_change = lambda state: send_json({"type": "status", "state": state.value})
    voice_pipeline.on_transcript = lambda text, is_final: send_json({"type": "transcript", "text": text, "is_final": is_final})
    voice_pipeline.on_text_delta = lambda delta: send_json({"type": "text_delta", "content": delta})
    voice_pipeline.on_latency = lambda metrics: send_json({"type": "latency", "metrics": metrics})

    # Notify connected
    await send_json({"type": "status", "state": AssistantVoiceState.IDLE.value})

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                # Binary PCM audio frame
                pcm_frame = message["bytes"]
                async for audio_chunk in voice_pipeline.process_incoming_pcm_frame(pcm_frame):
                    await send_audio(audio_chunk)

            elif "text" in message and message["text"]:
                # Control / JSON frame
                try:
                    payload = json.loads(message["text"])
                    cmd_type = payload.get("type")

                    if cmd_type == "ptt_start":
                        await voice_pipeline.trigger_push_to_talk_start()

                    elif cmd_type == "ptt_stop":
                        async for audio_chunk in voice_pipeline.trigger_push_to_talk_stop():
                            await send_audio(audio_chunk)

                    elif cmd_type == "confirm":
                        action = payload.get("action")  # approve / deny
                        conf_id = payload.get("confirmation_id")
                        if confirmation_manager and conf_id:
                            if action == "approve":
                                confirmation_manager.approve(conf_id)
                                await send_json({"type": "confirmation_result", "status": "approved"})
                            else:
                                confirmation_manager.deny(conf_id)
                                await send_json({"type": "confirmation_result", "status": "denied"})

                    elif cmd_type == "barge_in":
                        voice_pipeline.duplex.check_barge_in(is_speech=True)

                    elif cmd_type == "reset":
                        voice_pipeline.reset_audio_buffer()
                        voice_pipeline._set_state(AssistantVoiceState.IDLE)

                except json.JSONDecodeError:
                    logger.debug("Received invalid JSON on voice WS: %s", message["text"])

    except WebSocketDisconnect:
        logger.info("Voice WebSocket client disconnected.")
    except Exception as e:
        logger.error("Error in voice WebSocket session: %s", e)
    finally:
        voice_pipeline.reset_audio_buffer()
