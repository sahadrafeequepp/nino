"""WebSocket endpoint /ws/avatar for live avatar animation, emotion, and speech lip-sync."""

import asyncio
import json
import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel

from nino.config import Settings, get_settings
from nino.persona.emotions import EmotionParser
from nino.persona.manager import PersonaManager
from nino.voice.duplex import AssistantVoiceState
from nino.voice.pipeline import VoicePipeline

logger = logging.getLogger("nino.api.avatar_ws")

router = APIRouter(tags=["avatar"])


class SetModeRequest(BaseModel):
    mode: str


class PersonaStatusResponse(BaseModel):
    name: str
    current_mode: str
    available_modes: list[str]


def verify_ws_token(token: Optional[str], settings: Settings) -> bool:
    """Verify Bearer token in constant time for WebSocket connections."""
    if not token:
        return False
    if token.startswith("Bearer "):
        token = token[7:]
    return secrets.compare_digest(token.encode("utf-8"), settings.auth_token.encode("utf-8"))


@router.get("/persona", response_model=PersonaStatusResponse, summary="Get persona and active mode")
async def get_persona_status(request: Request, settings: Settings = Depends(get_settings)):
    pm: PersonaManager = getattr(request.app.state, "persona_manager", None) or PersonaManager()
    return PersonaStatusResponse(
        name=pm.name,
        current_mode=pm.get_mode(),
        available_modes=list(pm.data.get("modes", {}).keys()),
    )


@router.post("/persona/mode", summary="Switch persona interaction mode")
async def set_persona_mode(request: Request, payload: SetModeRequest, settings: Settings = Depends(get_settings)):
    pm: PersonaManager = getattr(request.app.state, "persona_manager", None) or PersonaManager()
    success = pm.set_mode(payload.mode)
    if not success:
        return {"status": "error", "message": f"Invalid mode '{payload.mode}'"}
    return {"status": "ok", "mode": pm.get_mode()}


@router.websocket("/ws/avatar")
async def avatar_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    settings: Settings = Depends(get_settings),
):
    """WebSocket stream for avatar animation state, emotion blendshapes, audio lip-sync, and chat."""
    # 1. Authenticate connection
    auth_header = websocket.headers.get("Authorization")
    extracted_token = token
    if not extracted_token and auth_header and auth_header.startswith("Bearer "):
        extracted_token = auth_header[7:]

    if not verify_ws_token(extracted_token, settings):
        logger.warning("Rejecting unauthenticated avatar WebSocket connection.")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info("Avatar WebSocket client connected.")

    app_state = websocket.app.state
    persona_mgr: PersonaManager = getattr(app_state, "persona_manager", None) or PersonaManager()
    voice_pipeline: VoicePipeline = getattr(app_state, "voice_pipeline", None)
    conv_store = getattr(app_state, "conv_store", None)
    adapter = getattr(app_state, "adapter", None)
    retrieval = getattr(app_state, "retrieval", None)
    orchestrator = getattr(app_state, "tool_orchestrator", None)

    emotion_parser = EmotionParser()

    async def send_json(data: dict):
        try:
            await websocket.send_text(json.dumps(data))
        except Exception:
            pass

    async def send_bytes(data: bytes):
        try:
            await websocket.send_bytes(data)
        except Exception:
            pass

    # Initial state handshake
    await send_json({
        "type": "state",
        "state": AssistantVoiceState.IDLE.value,
    })
    await send_json({
        "type": "mode",
        "mode": persona_mgr.get_mode(),
    })
    await send_json({
        "type": "emotion",
        "emotion": "neutral",
    })

    # Hook voice pipeline events if present
    if voice_pipeline:
        orig_state_change = voice_pipeline.on_state_change
        orig_transcript = voice_pipeline.on_transcript
        orig_text_delta = voice_pipeline.on_text_delta
        orig_latency = voice_pipeline.on_latency

        async def avatar_on_state(state: AssistantVoiceState):
            await send_json({"type": "state", "state": state.value})
            if orig_state_change:
                res = orig_state_change(state)
                if asyncio.iscoroutine(res):
                    await res

        async def avatar_on_transcript(text: str, is_final: bool):
            await send_json({"type": "transcript", "text": text, "is_final": is_final})
            if orig_transcript:
                res = orig_transcript(text, is_final)
                if asyncio.iscoroutine(res):
                    await res

        async def avatar_on_delta(chunk: str):
            clean_text, emotions = emotion_parser.process_chunk(chunk)
            for emo in emotions:
                await send_json({"type": "emotion", "emotion": emo})
            if clean_text:
                await send_json({"type": "text_delta", "content": clean_text})
            if orig_text_delta:
                res = orig_text_delta(clean_text)
                if asyncio.iscoroutine(res):
                    await res

        async def avatar_on_latency(metrics: dict):
            await send_json({"type": "latency", "metrics": metrics})
            if orig_latency:
                res = orig_latency(metrics)
                if asyncio.iscoroutine(res):
                    await res

        voice_pipeline.on_state_change = avatar_on_state
        voice_pipeline.on_transcript = avatar_on_transcript
        voice_pipeline.on_text_delta = avatar_on_delta
        voice_pipeline.on_latency = avatar_on_latency

    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"]:
                # Stream microphone audio frame to voice pipeline
                if voice_pipeline:
                    async for audio_chunk in voice_pipeline.process_incoming_pcm_frame(msg["bytes"]):
                        await send_bytes(audio_chunk)

            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    cmd_type = data.get("type")

                    if cmd_type == "set_mode":
                        new_mode = data.get("mode", "companion")
                        if persona_mgr.set_mode(new_mode):
                            await send_json({"type": "mode", "mode": persona_mgr.get_mode()})
                            await send_json({"type": "mode_changed", "mode": persona_mgr.get_mode()})

                    elif cmd_type == "ptt_start" and voice_pipeline:
                        await voice_pipeline.trigger_push_to_talk_start()

                    elif cmd_type == "ptt_stop" and voice_pipeline:
                        async for audio_chunk in voice_pipeline.trigger_push_to_talk_stop():
                            await send_bytes(audio_chunk)

                    elif cmd_type in ("chat", "user_text"):
                        user_text = data.get("text", "").strip()
                        if not user_text:
                            continue

                        # Check for mode switch command in text
                        detected_mode = persona_mgr.detect_mode_switch_command(user_text)
                        if detected_mode:
                            persona_mgr.set_mode(detected_mode)
                            await send_json({"type": "mode", "mode": persona_mgr.get_mode()})
                            await send_json({"type": "mode_changed", "mode": persona_mgr.get_mode()})

                        # Check crisis turn
                        is_crisis = persona_mgr.is_crisis_turn(user_text)
                        if is_crisis:
                            crisis_text = persona_mgr.get_crisis_response()
                            await send_json({"type": "state", "state": "speaking"})
                            await send_json({"type": "emotion", "emotion": "concerned"})
                            await send_json({"type": "delta", "text": crisis_text})
                            await send_json({"type": "text_delta", "content": crisis_text})
                            await send_json({"type": "text_complete", "emotion": "concerned"})
                            await send_json({"type": "state", "state": "idle"})
                            continue

                        # Process chat turn
                        await send_json({"type": "state", "state": "thinking"})
                        session_id = conv_store.ensure_session(None) if conv_store else "avatar-session"
                        
                        retrieved_ctx = None
                        if retrieval and not is_crisis:
                            retrieved_ctx = await retrieval.retrieve_context(user_text)

                        system_content = persona_mgr.build_system_prompt(
                            retrieved_memory_prompt=retrieved_ctx.system_prompt if retrieved_ctx else None
                        )

                        messages = [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_text},
                        ]

                        emotion_parser.reset()
                        await send_json({"type": "state", "state": "speaking"})

                        full_reply = []
                        last_emotion = "neutral"
                        tts_adapter = voice_pipeline.tts if voice_pipeline else None

                        if adapter:
                            from nino.voice.streamer import SentenceBoundarySplitter
                            splitter = SentenceBoundarySplitter()

                            async for chunk in adapter.stream_chat(messages, model=settings.model_name):
                                clean_chunk, emotions = emotion_parser.process_chunk(chunk)
                                for emo in emotions:
                                    last_emotion = emo
                                    await send_json({"type": "emotion", "emotion": emo})
                                if clean_chunk:
                                    full_reply.append(clean_chunk)
                                    await send_json({"type": "delta", "text": clean_chunk})
                                    await send_json({"type": "text_delta", "content": clean_chunk})

                                if tts_adapter:
                                    for sentence in splitter.process_chunk(clean_chunk):
                                        audio_pcm = await tts_adapter.synthesize(sentence)
                                        if audio_pcm:
                                            await send_bytes(audio_pcm)

                            # Flush emotion and sentence buffers
                            leftover_text, final_emotions = emotion_parser.flush()
                            for emo in final_emotions:
                                last_emotion = emo
                                await send_json({"type": "emotion", "emotion": emo})
                            if leftover_text:
                                full_reply.append(leftover_text)
                                await send_json({"type": "delta", "text": leftover_text})
                                await send_json({"type": "text_delta", "content": leftover_text})

                            if tts_adapter:
                                for sentence in splitter.flush():
                                    audio_pcm = await tts_adapter.synthesize(sentence)
                                    if audio_pcm:
                                        await send_bytes(audio_pcm)

                        final_reply = "".join(full_reply).strip()
                        if conv_store and final_reply:
                            conv_store.add_message(session_id, "user", user_text)
                            conv_store.add_message(session_id, "assistant", final_reply)

                        await send_json({"type": "text_complete", "emotion": last_emotion})
                        await send_json({"type": "state", "state": "idle"})

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("Avatar WebSocket client disconnected.")
    except Exception as e:
        logger.error("Error in avatar WebSocket session: %s", e)
