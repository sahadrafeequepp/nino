"""Voice interaction pipeline coordinating Wake, VAD, STT, LLM Streaming, TTS, and Confirmation."""

import asyncio
import io
import json
import logging
from typing import Any, AsyncIterator, Callable, Optional

from nino.config import Settings
from nino.memory.conversation import ConversationStore
from nino.memory.facts import FactManager
from nino.memory.retrieval import RetrievalPipeline
from nino.models.base import LLMAdapter
from nino.tools import Confirmation, ConfirmationManager, ToolOrchestrator
from nino.voice.confirmation_flow import SpokenConfirmationHandler
from nino.voice.duplex import AssistantVoiceState, DuplexController, DuplexMode
from nino.voice.interfaces import STTAdapter, TTSAdapter, VADDetector, WakeWordDetector
from nino.voice.streamer import SentenceBoundarySplitter
from nino.voice.telemetry import VoiceLatencyTracker

logger = logging.getLogger("nino.voice.pipeline")


class VoicePipeline:
    """Core on-device voice processing pipeline."""

    def __init__(
        self,
        stt: STTAdapter,
        tts: TTSAdapter,
        wake_word: WakeWordDetector,
        vad: VADDetector,
        conv_store: ConversationStore,
        adapter: LLMAdapter,
        settings: Settings,
        retrieval: Optional[RetrievalPipeline] = None,
        orchestrator: Optional[ToolOrchestrator] = None,
        confirmation_manager: Optional[ConfirmationManager] = None,
        fact_manager: Optional[FactManager] = None,
        duplex_mode: DuplexMode = DuplexMode.HALF_DUPLEX,
    ):
        self.stt = stt
        self.tts = tts
        self.wake_word = wake_word
        self.vad = vad
        self.conv_store = conv_store
        self.adapter = adapter
        self.settings = settings
        self.retrieval = retrieval
        self.orchestrator = orchestrator
        self.confirmation_manager = confirmation_manager
        self.fact_manager = fact_manager

        self.duplex = DuplexController(mode=duplex_mode)
        self.confirmation_handler = (
            SpokenConfirmationHandler(confirmation_manager)
            if confirmation_manager
            else None
        )

        # Internal state
        self.is_recording = False
        self.speech_buffer = bytearray()
        self.consecutive_silence_ms = 0
        self.consecutive_speech_ms = 0
        self.active_session_id: Optional[str] = None
        self.pending_confirmation: Optional[Confirmation] = None

        # Callbacks for client/WS notifications
        self.on_state_change: Optional[Callable[[AssistantVoiceState], None]] = None
        self.on_transcript: Optional[Callable[[str, bool], None]] = None
        self.on_text_delta: Optional[Callable[[str], None]] = None
        self.on_latency: Optional[Callable[[dict[str, float]], None]] = None

    async def _emit_state(self, state: AssistantVoiceState) -> None:
        self.duplex.set_state(state)
        if self.on_state_change:
            res = self.on_state_change(state)
            if asyncio.iscoroutine(res):
                await res

    async def _emit_transcript(self, text: str, is_final: bool) -> None:
        if self.on_transcript:
            res = self.on_transcript(text, is_final)
            if asyncio.iscoroutine(res):
                await res

    async def _emit_text_delta(self, delta: str) -> None:
        if self.on_text_delta:
            res = self.on_text_delta(delta)
            if asyncio.iscoroutine(res):
                await res

    async def _emit_latency(self, metrics: dict[str, float]) -> None:
        if self.on_latency:
            res = self.on_latency(metrics)
            if asyncio.iscoroutine(res):
                await res

    def reset_audio_buffer(self) -> None:
        """Discard in-memory audio buffers strictly without saving to disk."""
        self.speech_buffer.clear()
        self.consecutive_silence_ms = 0
        self.consecutive_speech_ms = 0

    async def trigger_push_to_talk_start(self) -> None:
        """Explicit trigger for Push-to-Talk activation."""
        self.reset_audio_buffer()
        self.is_recording = True
        await self._emit_state(AssistantVoiceState.LISTENING)
        logger.info("Push-to-Talk activated: Listening...")

    async def trigger_push_to_talk_stop(self) -> AsyncIterator[bytes]:
        """Release Push-to-Talk and process buffered audio."""
        if not self.is_recording:
            return

        self.is_recording = False
        logger.info("Push-to-Talk released. Processing captured speech (%d bytes)...", len(self.speech_buffer))
        async for audio_chunk in self._process_speech_turn():
            yield audio_chunk

    async def process_incoming_pcm_frame(
        self,
        frame_pcm: bytes,
        frame_duration_ms: int = 32,
    ) -> AsyncIterator[bytes]:
        """Ingest a streaming PCM audio frame from microphone or WebSocket.

        Yields synthesized audio bytes if a full conversational turn is triggered and answered.
        """
        # 1. Gate microphone input based on duplex state
        if not self.duplex.can_process_microphone():
            return

        is_speech = self.vad.is_speech(frame_pcm)

        # 2. Check for experimental barge-in if currently speaking
        if self.duplex.check_barge_in(is_speech):
            self.reset_audio_buffer()
            self.is_recording = True
            await self._emit_state(AssistantVoiceState.LISTENING)
            return

        # 3. If not yet recording, check wake word
        if not self.is_recording:
            wake_detected, score, _ = self.wake_word.process_frame(frame_pcm)
            if wake_detected:
                logger.info("Wake word detected (score=%.2f). Starting listening session...", score)
                self.reset_audio_buffer()
                self.is_recording = True
                await self._emit_state(AssistantVoiceState.LISTENING)
            return

        # 4. We are actively recording speech
        self.speech_buffer.extend(frame_pcm)

        if is_speech:
            self.consecutive_speech_ms += frame_duration_ms
            self.consecutive_silence_ms = 0
        else:
            self.consecutive_silence_ms += frame_duration_ms

        # 5. Check if speech has ended (silence timeout reached after having spoken)
        min_speech_duration_ms = 250
        silence_timeout_ms = self.settings.vad_silence_timeout_ms

        if (
            self.consecutive_speech_ms >= min_speech_duration_ms
            and self.consecutive_silence_ms >= silence_timeout_ms
        ):
            logger.info(
                "End of speech detected (speech=%dms, silence=%dms).",
                self.consecutive_speech_ms,
                self.consecutive_silence_ms,
            )
            self.is_recording = False
            async for audio_chunk in self._process_speech_turn():
                yield audio_chunk

    async def _process_speech_turn(self) -> AsyncIterator[bytes]:
        """Transcribe speech buffer, query LLM, and stream synthesized voice response."""
        telemetry = VoiceLatencyTracker()
        telemetry.mark_speech_end()

        await self._emit_state(AssistantVoiceState.PROCESSING)
        raw_pcm = bytes(self.speech_buffer)
        self.reset_audio_buffer()

        if not raw_pcm:
            await self._emit_state(AssistantVoiceState.IDLE)
            return

        # 1. Transcribe speech in memory (faster-whisper)
        transcript = await self.stt.transcribe(raw_pcm)
        telemetry.mark_stt_done()

        if not transcript or not transcript.strip():
            logger.debug("Empty transcript, returning to idle.")
            await self._emit_state(AssistantVoiceState.IDLE)
            return

        logger.info("Voice Transcript: '%s'", transcript)
        await self._emit_transcript(transcript, True)

        # 2. Check if answering a pending spoken confirmation
        if self.pending_confirmation and self.confirmation_handler:
            async for audio_chunk in self._handle_spoken_confirmation(
                transcript, telemetry
            ):
                yield audio_chunk
            return

        # 3. Normal conversational turn
        session_id = self.conv_store.ensure_session(self.active_session_id)
        self.active_session_id = session_id

        # Retrieve RAG context if available
        retrieved_ctx = None
        if self.retrieval:
            retrieved_ctx = await self.retrieval.retrieve_context(transcript)

        # Build prompt
        base_system = (
            "You are Nino, a helpful, security-minded, local-first personal AI assistant. "
            "Speak smoothly, naturally, and warmly like an articulate human friend. "
            "Do NOT use artificial speech fillers (like 'hmm' or 'well...'), forced pauses, or fragmented sentences. "
            "Do NOT ask unnecessary follow-up questions unless clarification is strictly required to proceed."
        )
        system_parts = [base_system]
        if retrieved_ctx and retrieved_ctx.system_prompt:
            system_parts.append(retrieved_ctx.system_prompt)
        if self.orchestrator:
            system_parts.append(self.orchestrator.build_tool_system_prompt())

        assembled_messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        history = await self.conv_store.get_budgeted_messages(
            session_id=session_id,
            token_budget=self.settings.token_budget_total - self.settings.token_budget_memory,
            adapter=self.adapter,
        )
        assembled_messages.extend(history)
        assembled_messages.append({"role": "user", "content": transcript})

        user_msg_id = self.conv_store.add_message(session_id, "user", transcript)

        # 4. Stream LLM tokens, split sentences, and synthesize TTS in parallel
        splitter = SentenceBoundarySplitter()
        full_assistant_reply = []
        is_first_token = True
        is_first_audio = True

        await self._emit_state(AssistantVoiceState.SPEAKING)

        try:
            async for chunk in self.adapter.stream_chat(
                assembled_messages, model=self.settings.model_name
            ):
                if is_first_token:
                    telemetry.mark_llm_first_token()
                    is_first_token = False

                full_assistant_reply.append(chunk)

                # Process sentence boundary
                sentences = splitter.process_chunk(chunk)
                for sentence in sentences:
                    if is_first_audio:
                        telemetry.mark_first_sentence()

                    # Synthesize completed sentence
                    audio_bytes = await self.tts.synthesize(sentence)
                    if audio_bytes:
                        if is_first_audio:
                            telemetry.mark_first_audio()
                            is_first_audio = False
                        await self._emit_text_delta(sentence + " ")
                        yield audio_bytes

            # Flush remaining sentence buffer
            for sentence in splitter.flush():
                audio_bytes = await self.tts.synthesize(sentence)
                if audio_bytes:
                    if is_first_audio:
                        telemetry.mark_first_audio()
                        is_first_audio = False
                    await self._emit_text_delta(sentence + " ")
                    yield audio_bytes

        except Exception as e:
            logger.error("Error during voice LLM stream / TTS: %s", e)
            err_msg = "I encountered an error processing your request."
            err_audio = await self.tts.synthesize(err_msg)
            if err_audio:
                yield err_audio

        final_text = "".join(full_assistant_reply).strip()
        if final_text:
            self.conv_store.add_message(session_id, "assistant", final_text)
            if self.fact_manager:
                asyncio.create_task(
                    self.fact_manager.extract_facts_from_turn(
                        user_message=transcript,
                        source_message_id=user_msg_id,
                        adapter=self.adapter,
                    )
                )

        telemetry.mark_playback_done()
        telemetry.log_summary()
        await self._emit_latency(telemetry.compute_metrics())

        await self._emit_state(AssistantVoiceState.IDLE)

    async def _handle_spoken_confirmation(
        self,
        spoken_text: str,
        telemetry: VoiceLatencyTracker,
    ) -> AsyncIterator[bytes]:
        """Process spoken confirmation response for a pending sensitive tool."""
        if not self.pending_confirmation or not self.confirmation_handler:
            return

        conf_id = self.pending_confirmation.id
        is_approved, feedback = self.confirmation_handler.process_spoken_reply(
            confirmation_id=conf_id,
            spoken_text=spoken_text,
        )

        # Clear pending confirmation
        prev_conf = self.pending_confirmation
        self.pending_confirmation = None

        await self._emit_state(AssistantVoiceState.SPEAKING)

        # Speak feedback
        feedback_audio = await self.tts.synthesize(feedback)
        if feedback_audio:
            yield feedback_audio

        # If approved, execute the tool through the orchestrator
        if is_approved and self.orchestrator and self.active_session_id:
            try:
                # Re-run orchestrator with approved confirmation ID
                assembled_messages = [
                    {"role": "system", "content": "You are Nino. Summarize the tool result concisely."},
                    {"role": "user", "content": f"Execute confirmed action {prev_conf.tool_name}."},
                ]
                orch_res = await self.orchestrator.run(
                    messages=assembled_messages,
                    adapter=self.adapter,
                    confirmation_id=conf_id,
                    context_data=assembled_messages,
                )
                if orch_res.reply:
                    self.conv_store.add_message(self.active_session_id, "assistant", orch_res.reply)
                    tool_audio = await self.tts.synthesize(orch_res.reply)
                    if tool_audio:
                        yield tool_audio
            except Exception as e:
                logger.error("Error executing confirmed tool: %s", e)

        await self._emit_state(AssistantVoiceState.IDLE)

    def request_tool_confirmation(self, confirmation: Confirmation) -> Optional[str]:
        """Set a pending confirmation and return the spoken prompt."""
        self.pending_confirmation = confirmation
        if self.confirmation_handler:
            return self.confirmation_handler.build_spoken_prompt(confirmation)
        return f"Nino needs approval to execute {confirmation.tool_name}. Say yes or no."
