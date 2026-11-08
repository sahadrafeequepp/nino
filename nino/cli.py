"""Interactive streaming CLI client and memory administration tool for Nino."""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any
import httpx
import logging
import threading

# Silence noisy HTTP access logs so only clean assistant output and thinking dots appear
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").disabled = True

from nino.voice.streamer import SentenceBoundarySplitter


from nino.config import get_settings
from nino.memory.db import MemoryDB
from nino.memory.ingest import DocumentIngestionPipeline
from nino.memory.vector.ollama_embedder import OllamaEmbedder
from nino.memory.vector.sqlite_store import SQLiteVectorStore

# Ensure Windows terminal handles UTF-8 emojis and special characters cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_synced_player = None


def get_synced_player(sample_rate: int = 22050):
    global _synced_player
    if _synced_player is None or getattr(_synced_player, "sample_rate", None) != sample_rate:
        from nino.voice.synced_player import SyncedAudioPlayer
        _synced_player = SyncedAudioPlayer(sample_rate=sample_rate)
    return _synced_player


async def speak_and_print_sentence(sentence_text: str, tts_adapter: Any, player: Any = None):
    """Play audio for a sentence and print words in real time across the audio duration."""
    if not sentence_text or not sentence_text.strip():
        return

    # Synthesize audio on worker thread
    audio_bytes = await asyncio.to_thread(tts_adapter._synthesize_sync, sentence_text)
    if not audio_bytes:
        sys.stdout.write(sentence_text.strip() + " ")
        sys.stdout.flush()
        return

    active_player = player or get_synced_player(sample_rate=tts_adapter.sample_rate)
    await active_player.play_and_print_async(sentence_text, audio_bytes)


async def speak_and_sync_print(full_text: str, tts_adapter: Any, player: Any = None):
    """Split a full text block and speak/print each sentence in sync."""
    splitter = SentenceBoundarySplitter()
    sentences = splitter.process_chunk(full_text) + splitter.flush()
    if not sentences and full_text.strip():
        sentences = [full_text.strip()]
    for s in sentences:
        await speak_and_print_sentence(s, tts_adapter, player)
        await asyncio.sleep(0.1)
    print()


async def stream_chat(
    base_url: str,
    token: str,
    messages: list[dict[str, str]],
    session_id: str | None = None,
    model: str | None = None,
    tts_adapter: Any = None,
    player: Any = None,
) -> tuple[str, str | None, list[str]]:
    """Send conversation messages to the streaming endpoint and yield tokens in real-time.

    If tts_adapter is provided, speech audio and displayed text are synced in lockstep:
    sentences are synthesized and words are typed in real time as the audio plays.
    """
    url = f"{base_url.rstrip('/')}/chat/stream"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload: dict = {"messages": messages}
    if session_id:
        payload["session_id"] = session_id
    if model:
        payload["model"] = model

    full_reply: list[str] = []
    received_session_id = session_id
    citations: list[str] = []
    splitter = SentenceBoundarySplitter() if tts_adapter else None

    sentence_queue: asyncio.Queue = asyncio.Queue() if tts_adapter else None
    active_player = player or (get_synced_player(sample_rate=tts_adapter.sample_rate) if tts_adapter else None)

    async def _speech_playback_worker():
        while True:
            item = await sentence_queue.get()
            if item is None:
                sentence_queue.task_done()
                break
            sent_text, synth_task = item
            try:
                audio_bytes = await synth_task
                if audio_bytes and active_player:
                    await active_player.play_and_print_async(sent_text, audio_bytes)
                    await asyncio.sleep(0.1)
                else:
                    sys.stdout.write(sent_text.strip() + " ")
                    sys.stdout.flush()
                    await asyncio.sleep(0.3)
            except Exception:
                sys.stdout.write(sent_text.strip() + " ")
                sys.stdout.flush()
            finally:
                sentence_queue.task_done()

    worker_task = asyncio.create_task(_speech_playback_worker()) if tts_adapter else None

    from nino.voice.synced_player import ThinkingIndicator
    thinking = ThinkingIndicator(prefix="Nino: ")
    thinking.start()
    is_first_chunk = True

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code == 401:
                        thinking.stop(clear_line=True)
                        print("\n[Error 401] Authentication failed: invalid or missing Bearer token.", file=sys.stderr)
                        return "", session_id, []
                    elif response.status_code != 200:
                        thinking.stop(clear_line=True)
                        err_body = await response.aread()
                        print(f"\n[Error {response.status_code}] {err_body.decode(errors='replace')}", file=sys.stderr)
                        return "", session_id, []

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("event: meta"):
                            continue
                        if line.startswith("data: "):
                            raw_data = line[6:].strip()
                            if raw_data == "[DONE]":
                                break
                            try:
                                payload_data = json.loads(raw_data)
                                if "session_id" in payload_data and payload_data["session_id"]:
                                    received_session_id = payload_data["session_id"]
                                    citations = payload_data.get("citations", [])

                                if "content" in payload_data and payload_data["content"]:
                                    chunk = payload_data["content"]
                                    full_reply.append(chunk)

                                    if is_first_chunk:
                                        thinking.stop(clear_line=True)
                                        sys.stdout.write("Nino: ")
                                        sys.stdout.flush()
                                        is_first_chunk = False

                                    if splitter and tts_adapter and sentence_queue:
                                        sentences = splitter.process_chunk(chunk)
                                        for s in sentences:
                                            synth_task = asyncio.create_task(asyncio.to_thread(tts_adapter._synthesize_sync, s))
                                            await sentence_queue.put((s, synth_task))
                                    else:
                                        sys.stdout.write(chunk)
                                        sys.stdout.flush()

                                elif "error" in payload_data:
                                    thinking.stop(clear_line=True)
                                    print(f"\n[Server Error]: {payload_data['error']}", file=sys.stderr)
                            except json.JSONDecodeError:
                                continue

                    # If voice is enabled, flush remaining text and wait for playback to finish
                    if splitter and tts_adapter and sentence_queue and worker_task:
                        for s in splitter.flush():
                            synth_task = asyncio.create_task(asyncio.to_thread(tts_adapter._synthesize_sync, s))
                            await sentence_queue.put((s, synth_task))
                        await sentence_queue.put(None)
                        await worker_task

            except httpx.ConnectError:
                thinking.stop(clear_line=True)
                print(f"\n[Connection Error] Could not connect to assistant at {base_url}. Is the server running?", file=sys.stderr)
                return "", session_id, []
            except Exception as e:
                thinking.stop(clear_line=True)
                print(f"\n[Unexpected Error]: {e}", file=sys.stderr)
                return "", session_id, []
    finally:
        thinking.stop(clear_line=False)
        if worker_task and not worker_task.done():
            try:
                await sentence_queue.put(None)
                await worker_task
            except Exception:
                pass

    print()
    if citations:
        print(f"\n[Sources cited: {', '.join(citations)}]")
    final_session_id = received_session_id or session_id
    return "".join(full_reply), final_session_id, citations


async def send_chat(
    base_url: str,
    token: str,
    messages: list[dict[str, str]],
    session_id: str | None = None,
    model: str | None = None,
    confirmation_id: str | None = None,
) -> dict[str, Any]:
    """Send chat request to the main chat completion endpoint."""
    url = f"{base_url.rstrip('/')}/chat"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload: dict = {"messages": messages}
    if session_id:
        payload["session_id"] = session_id
    if model:
        payload["model"] = model
    if confirmation_id:
        payload["confirmation_id"] = confirmation_id

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


TOOL_TRIGGERS = {
    "task", "reminder", "calendar", "event", "note", "save", "delete",
    "forget", "calculate", "search", "who am i", "remember", "schedule",
    "subagent", "delegate", "agent", "weather", "forecast", "temperature",
    "location", "gps", "where am i", "rain", "climate", "outside",
    "time", "clock", "date", "day", "samayam", "ethrayaayi", "tell me the time",
}


async def chat_interactive(args: argparse.Namespace) -> None:
    """Run clean, interactive terminal session with real-time streaming and tool support."""
    settings = get_settings()
    target_host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    base_url = args.url or f"http://{target_host}:{settings.port}"
    token = args.token or settings.auth_token
    model_name = args.model or settings.model_name
    stream_mode = getattr(args, "stream", True)

    print("=" * 66)
    print(f"  Nino AI Assistant ({model_name}) • Local-First & Memory-Persistent")
    print("  Commands: '/sessions' list | '/load <id>' switch | '/save <name>'")
    print("            '/remember <info>' store | '/memories' view | '/help' all")
    print("=" * 66)

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/health")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ollama_reachable") is False:
                    print("  [Notice] Ollama model backend appears offline.")
    except Exception:
        print(f"  [Warning] Backend not detected at {base_url}. Start server with: python -m uvicorn nino.main:app")

    session_id = getattr(args, "session", None)
    conversation: list[dict[str, str]] = []

    # Voice / Speech output configuration
    speak_enabled = getattr(args, "speak", False)
    voice_choice = getattr(args, "tts_voice", None) or settings.tts_voice
    tts_adapter = None
    player = None
    if speak_enabled:
        try:
            from nino.voice.tts import PiperTTSAdapter, resolve_voice_name, get_voice_info
            from nino.voice.synced_player import SyncedAudioPlayer
            resolved_voice = resolve_voice_name(voice_choice)
            v_info = get_voice_info(resolved_voice)
            tts_adapter = PiperTTSAdapter(voice=resolved_voice, speed=settings.tts_speed)
            player = SyncedAudioPlayer(sample_rate=tts_adapter.sample_rate)
            print(f"  [🔊 Voice output enabled: '{v_info['name']}' ({resolved_voice}) - {v_info.get('accent', '')}]")
        except Exception as e:
            print(f"  [Notice: Could not initialize voice synthesizer: {e}]")

    # Auto-resume prior session if not explicitly starting fresh
    if not getattr(args, "new", False) and not getattr(args, "no_resume", False) and not session_id:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(
                    f"{base_url.rstrip('/')}/memory/sessions/latest",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if res.status_code == 200:
                    latest = res.json()
                    sess_data = latest.get("session")
                    msgs = latest.get("messages", [])
                    if sess_data and msgs:
                        session_id = sess_data["id"]
                        title = sess_data.get("title") or "Untitled Session"
                        for m in msgs:
                            conversation.append({"role": m["role"], "content": m["content"]})
                        print(f"  [Resumed prior session: '{title}' ({len(conversation)} messages restored)]\n")
        except Exception:
            pass
    elif session_id:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.get(
                    f"{base_url.rstrip('/')}/memory/sessions/{session_id}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                if res.status_code == 200:
                    data = res.json()
                    sess_data = data.get("session", {})
                    title = sess_data.get("title", session_id) if sess_data else session_id
                    for m in data.get("messages", []):
                        conversation.append({"role": m["role"], "content": m["content"]})
                    print(f"  [Loaded session: '{title}' ({len(conversation)} messages restored)]\n")
        except Exception as e:
            print(f"  [Warning: Could not load session {session_id}: {e}]")

    while True:
        try:
            sys.stdout.write("\nYou: ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            if not user_input:
                break
            user_input = user_input.strip()

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in ("/exit", "/quit", "exit", "quit", ":q"):
                print("Goodbye!")
                break

            if lowered in ("/clear", "clear", ":c"):
                conversation.clear()
                print("[Conversation view cleared]")
                continue

            if lowered in ("/help", "help", ":h", "?"):
                print("\n  Nino Local Memory & Assistant Commands:")
                print("    /voices                : List all available voices (female, male, accents)")
                print("    /voice <name_or_num>   : Switch voice live (e.g. /voice kristin, /voice cori, /voice ryan)")
                print("    /sessions, /list       : List all saved local sessions in SQLite")
                print("    /load <id_or_number>   : Load prior session and restore all context")
                print("    /save <title>          : Name and save current conversation")
                print("    /new                   : Start a fresh conversation session")
                print("    /remember <info>       : Store a durable personal memory permanently")
                print("    /memories, /facts      : View all permanent facts stored in SQLite")
                print("    /forget <query>        : Semantic search and delete facts")
                print("    /mode                  : Toggle between fast streaming and tool orchestrator")
                print("    /clear                 : Clear the current screen view")
                print("    /exit, /quit           : Exit Nino CLI\n")
                continue

            if lowered in ("/voices", "voices") or (lowered in ("/voice", "voice") and not user_input.strip().startswith(("/voice ", "voice "))):
                from nino.voice.tts import VOICE_CATALOG, is_voice_downloaded
                current_id = tts_adapter.voice if tts_adapter else (getattr(args, "tts_voice", None) or settings.tts_voice)
                print("\n  Available Nino Voices (100% On-Device Neural Piper TTS):")
                print("  " + "=" * 76)
                print(f"  {'#':<3} {'Alias':<10} {'Gender':<8} {'Accent':<18} {'Style / Description':<32}")
                print("  " + "-" * 76)
                for idx, (alias, info) in enumerate(VOICE_CATALOG.items(), 1):
                    is_active = (info["id"] == current_id or alias == current_id)
                    marker = " * " if is_active else "   "
                    status = " [Ready]" if is_voice_downloaded(alias) else " [On-Demand]"
                    print(f" {marker}[{idx}] {info['name']:<10} {info['gender']:<8} {info['accent']:<18} {info['style']}{status}")
                print("  " + "=" * 76)
                print(f"  Active Voice : {current_id} ({'🔊 Enabled' if speak_enabled else 'Muted - use /voice <name> to enable'})")
                print("  Switch Live  : /voice <alias_or_number>  (e.g., /voice kristin, /voice cori, /voice ryan)")
                print("  Permanent    : Set TTS_VOICE=en_US-kristin-medium in your .env file\n")
                continue

            if lowered.startswith("/voice ") or lowered.startswith("voice "):
                voice_target = user_input.split(" ", 1)[1].strip()
                from nino.voice.tts import PiperTTSAdapter, resolve_voice_name, get_voice_info
                from nino.voice.synced_player import SyncedAudioPlayer

                resolved = resolve_voice_name(voice_target)
                v_info = get_voice_info(resolved)

                try:
                    print(f"  [*] Loading voice '{v_info['name']}' ({resolved})...")
                    new_adapter = PiperTTSAdapter(voice=resolved, speed=settings.tts_speed)
                    new_adapter.download_voice_if_needed()
                    new_player = SyncedAudioPlayer(sample_rate=new_adapter.sample_rate)

                    tts_adapter = new_adapter
                    player = new_player
                    speak_enabled = True

                    print(f"  [🔊 Voice switched to: '{v_info['name']}' ({resolved}) - {v_info.get('accent', '')}]")
                    print(f"  [Tip: Set TTS_VOICE={resolved} in .env to keep it as your permanent default]\n")

                    sample_text = f"Hello! This is my new {v_info['name']} voice. How does it sound to you?"
                    def _preview_sample():
                        try:
                            for chunk, p_ms in new_adapter.synthesize_stream(sample_text):
                                new_player.play_chunk(chunk)
                                if p_ms > 0:
                                    import time
                                    time.sleep(p_ms / 1000.0)
                        except Exception:
                            pass
                    import threading
                    threading.Thread(target=_preview_sample, daemon=True).start()
                except Exception as e:
                    print(f"  [Error switching voice: {e}]")
                continue

            if lowered in ("/sessions", "/list", "sessions", "list"):
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        res = await client.get(
                            f"{base_url.rstrip('/')}/memory/sessions",
                            headers={"Authorization": f"Bearer {token}"},
                        )
                        if res.status_code == 200:
                            s_list = res.json().get("sessions", [])
                            if not s_list:
                                print("  [No saved sessions found in local SQLite database.]")
                            else:
                                print("\n  Saved Local Sessions:")
                                print("  " + "-" * 70)
                                for idx, s in enumerate(s_list, 1):
                                    active_marker = " * " if s["id"] == session_id else "   "
                                    title = s.get("title") or "Untitled Session"
                                    time_str = s.get("updated_at", "")[:16]
                                    count = s.get("message_count", 0)
                                    sid_short = s["id"][:8]
                                    print(f"  {active_marker}[{idx}] {sid_short}... | {time_str} | {count:2d} msgs | {title}")
                                print("  " + "-" * 70)
                                print("  Use '/load <number_or_id>' to switch to any session.\n")
                        else:
                            print(f"  [Failed to list sessions: {res.status_code}]")
                except Exception as e:
                    print(f"  [Error contacting memory store: {e}]")
                continue

            if lowered.startswith("/load ") or lowered.startswith("load "):
                load_target = user_input.split(" ", 1)[1].strip()
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        target_id = load_target
                        if load_target.isdigit():
                            idx = int(load_target) - 1
                            s_res = await client.get(
                                f"{base_url.rstrip('/')}/memory/sessions",
                                headers={"Authorization": f"Bearer {token}"},
                            )
                            if s_res.status_code == 200:
                                s_list = s_res.json().get("sessions", [])
                                if 0 <= idx < len(s_list):
                                    target_id = s_list[idx]["id"]

                        res = await client.get(
                            f"{base_url.rstrip('/')}/memory/sessions/{target_id}",
                            headers={"Authorization": f"Bearer {token}"},
                        )
                        if res.status_code == 200:
                            data = res.json()
                            sess_data = data.get("session", {})
                            msgs = data.get("messages", [])
                            session_id = target_id
                            conversation.clear()
                            for m in msgs:
                                conversation.append({"role": m["role"], "content": m["content"]})
                            title = sess_data.get("title", target_id)
                            print(f"\n  [Loaded session: '{title}' ({len(conversation)} messages restored)]")
                            if conversation:
                                print("  Recent context:")
                                for m in conversation[-3:]:
                                    prefix = "You" if m["role"] == "user" else "Nino"
                                    snippet = m["content"][:100] + ("..." if len(m["content"]) > 100 else "")
                                    print(f"    {prefix}: {snippet}")
                                print()
                        else:
                            print(f"  [Session '{load_target}' not found.]")
                except Exception as e:
                    print(f"  [Error loading session: {e}]")
                continue

            if lowered.startswith("/save ") or lowered.startswith("/store ") or lowered.startswith("save "):
                save_title = user_input.split(" ", 1)[1].strip()
                if not session_id:
                    session_id = str(uuid.uuid4())
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        res = await client.post(
                            f"{base_url.rstrip('/')}/memory/sessions/{session_id}/title",
                            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                            json={"title": save_title},
                        )
                        if res.status_code == 200:
                            print(f"  [Session successfully stored and named: '{save_title}']")
                        else:
                            print(f"  [Failed to save session title: {res.status_code}]")
                except Exception as e:
                    print(f"  [Error saving session: {e}]")
                continue

            if lowered in ("/new", "new"):
                session_id = None
                conversation.clear()
                print("  [Started new clean session. Prior sessions remain stored in local SQLite.]")
                continue

            if lowered.startswith("/remember ") or lowered.startswith("remember "):
                fact_text = user_input.split(" ", 1)[1].strip()
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        res = await client.post(
                            f"{base_url.rstrip('/')}/memory/facts/store",
                            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                            json={"text": fact_text, "category": "general"},
                        )
                        if res.status_code == 200:
                            data = res.json()
                            if data.get("status") == "saved":
                                print(f"  [Memory Stored: '{fact_text}' — Persisted locally in SQLite & vector store!]")
                            else:
                                print("  [Notice: Fact was already remembered or filtered as sensitive]")
                        else:
                            print(f"  [Failed to store fact: {res.status_code}]")
                except Exception as e:
                    print(f"  [Error storing memory: {e}]")
                continue

            if lowered in ("/memories", "/facts", "memories", "facts"):
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        res = await client.get(
                            f"{base_url.rstrip('/')}/memory/facts",
                            headers={"Authorization": f"Bearer {token}"},
                        )
                        if res.status_code == 200:
                            facts = res.json().get("facts", [])
                            if not facts:
                                print("  [No stored facts yet. Use '/remember <fact>' to store personal info!]")
                            else:
                                print("\n  Permanent Stored Facts (SQLite + Vector Store):")
                                print("  " + "-" * 60)
                                for f in facts:
                                    cat = f.get("category", "general")
                                    text = f.get("text", "")
                                    print(f"  • [{cat.upper()}] {text}")
                                print("  " + "-" * 60 + "\n")
                        else:
                            print(f"  [Failed to retrieve memories: {res.status_code}]")
                except Exception as e:
                    print(f"  [Error retrieving memories: {e}]")
                continue

            if lowered in ("/mode", "/toggle", "/stream"):
                stream_mode = not stream_mode
                mode_str = "Streaming (fast direct response)" if stream_mode else "Tool Orchestrator (actions enabled)"
                print(f"[Mode switched to: {mode_str}]")
                continue

            conversation.append({"role": "user", "content": user_input})

            # Fast direct response for live time/date queries
            try:
                from telegram_bot.clock import is_time_query, get_time_conversational_reply
                if is_time_query(user_input):
                    time_reply = get_time_conversational_reply(user_input)
                    conversation.append({"role": "assistant", "content": time_reply})
                    sys.stdout.write(f"\n{time_reply}\n")
                    sys.stdout.flush()
                    if speak_enabled and tts_adapter and player:
                        await tts_adapter.synthesize_and_play(time_reply, player=player)
                    continue
            except Exception:
                pass

            # Check whether this prompt requires multi-turn tool calling
            is_tool_intent = (
                not stream_mode
                or lowered.startswith("/tool")
                or any(keyword in lowered for keyword in TOOL_TRIGGERS)
            )

            # 1. Direct Streaming Path (Conversational, coding, questions)
            if stream_mode and not is_tool_intent:
                sys.stdout.write("\n")
                sys.stdout.flush()
                try:
                    reply, new_session_id, citations = await stream_chat(
                        base_url=base_url,
                        token=token,
                        messages=conversation,
                        session_id=session_id,
                        model=args.model,
                        tts_adapter=tts_adapter if speak_enabled else None,
                        player=player if speak_enabled else None,
                    )
                    if new_session_id:
                        session_id = new_session_id
                    if reply:
                        conversation.append({"role": "assistant", "content": reply})
                    continue
                except httpx.ConnectError:
                    print(f"\n[Connection Error] Could not connect to assistant at {base_url}.", file=sys.stderr)
                    continue
                except httpx.TimeoutException:
                    print(f"\n[Timeout Error] Streaming timed out. Retrying with full orchestrator...", file=sys.stderr)
                    # Fall through to send_chat

            # 2. Tool Orchestrator Path (Actions, tasks, notes, or tool mode)
            clean_prompt = user_input[6:].strip() if lowered.startswith("/tool ") else user_input
            if clean_prompt != user_input:
                conversation[-1]["content"] = clean_prompt

            sys.stdout.write("\n")
            sys.stdout.flush()
            from nino.voice.synced_player import ThinkingIndicator
            orch_thinking = ThinkingIndicator(prefix="Nino: ")
            orch_thinking.start()

            try:
                chat_resp = await send_chat(
                    base_url=base_url,
                    token=token,
                    messages=conversation,
                    session_id=session_id,
                    model=args.model,
                )
            except httpx.HTTPStatusError as e:
                orch_thinking.stop(clear_line=True)
                err_detail = e.response.text
                try:
                    err_json = e.response.json()
                    if isinstance(err_json, dict) and "detail" in err_json:
                        err_detail = err_json["detail"]
                except Exception:
                    pass
                print(f"\n[Assistant Error {e.response.status_code}]: {err_detail}", file=sys.stderr)
                continue

            except httpx.ConnectError:
                orch_thinking.stop(clear_line=True)
                print(f"\n[Connection Error] Could not connect to assistant at {base_url}.", file=sys.stderr)
                continue
            except httpx.TimeoutException:
                orch_thinking.stop(clear_line=True)
                print(f"\n[Timeout Error] Request timed out on local model. Please try again.", file=sys.stderr)
                continue
            finally:
                orch_thinking.stop(clear_line=True)

            sys.stdout.write("Nino: ")
            sys.stdout.flush()

            new_sid = chat_resp.get("session_id")
            if new_sid:
                session_id = new_sid
            reply = chat_resp.get("reply", "")
            pending_conf = chat_resp.get("pending_confirmation")
            citations = chat_resp.get("citations", [])

            if speak_enabled and tts_adapter:
                await speak_and_sync_print(reply, tts_adapter, player)
            else:
                print(f"{reply}")

            if citations:
                print(f"\n[Sources cited: {', '.join(citations)}]")

            # Handle interactive confirmation prompt
            if pending_conf:
                conf_id = pending_conf["id"]
                tool_name = pending_conf["tool_name"]
                tier = pending_conf["tier"]
                summary = pending_conf["summary"]
                args_dict = pending_conf["arguments"]

                print("\n" + "-" * 50)
                print(f"  [!] CONFIRMATION REQUIRED")
                print(f"  Tool:      {tool_name} [{tier}]")
                print(f"  Action:    {summary}")
                print(f"  Arguments: {args_dict}")
                print("-" * 50)
                sys.stdout.write("  Approve this action? (y/N): ")
                sys.stdout.flush()
                choice = sys.stdin.readline().strip().lower()

                headers = {"Authorization": f"Bearer {token}"}
                async with httpx.AsyncClient(timeout=30.0) as client:
                    if choice in ("y", "yes"):
                        approve_resp = await client.post(
                            f"{base_url}/confirmations/{conf_id}/approve",
                            headers=headers,
                        )
                        if approve_resp.status_code == 200:
                            print("  [Action Approved] Executing now...")
                            resumed_resp = await send_chat(
                                base_url=base_url,
                                token=token,
                                messages=conversation,
                                session_id=session_id,
                                model=args.model,
                                confirmation_id=conf_id,
                            )
                            resumed_reply = resumed_resp.get("reply", "")
                            print(f"\nNino: {resumed_reply}")
                            conversation.append({"role": "assistant", "content": resumed_reply})
                        else:
                            print(f"  [Approval Error]: {approve_resp.text}")
                    else:
                        await client.post(
                            f"{base_url}/confirmations/{conf_id}/deny",
                            headers=headers,
                        )
                        print("  [Action Denied] Operation cancelled.")

            elif reply:
                conversation.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye!")
            break
        finally:
            pass
    if player:
        player.close()


async def handle_confirmations(args: argparse.Namespace) -> None:
    """CLI subcommands for managing pending tool confirmations."""
    settings = get_settings()
    target_host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    base_url = (getattr(args, "url", None) or f"http://{target_host}:{settings.port}").rstrip("/")
    token = getattr(args, "token", None) or settings.auth_token
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        if args.conf_action == "list":
            resp = await client.get(f"{base_url}/confirmations", headers=headers)
            if resp.status_code == 200:
                confs = resp.json()
                if not confs:
                    print("No pending confirmations found.")
                    return
                print(f"--- Pending Confirmations ({len(confs)}) ---")
                for c in confs:
                    print(f"- ID: {c['id']}")
                    print(f"  Tool: {c['tool_name']} [{c['tier']}]")
                    print(f"  Summary: {c['summary']}")
                    print(f"  Arguments: {c['arguments']}")
                    print(f"  Expires: {c['expires_at']}\n")
            else:
                print(f"Failed to list confirmations: {resp.status_code} {resp.text}")

        elif args.conf_action == "approve":
            resp = await client.post(f"{base_url}/confirmations/{args.id}/approve", headers=headers)
            if resp.status_code == 200:
                print(f"Confirmation '{args.id}' approved successfully.")
            else:
                print(f"Failed to approve: {resp.status_code} {resp.text}")

        elif args.conf_action == "deny":
            resp = await client.post(f"{base_url}/confirmations/{args.id}/deny", headers=headers)
            if resp.status_code == 200:
                print(f"Confirmation '{args.id}' denied successfully.")
            else:
                print(f"Failed to deny: {resp.status_code} {resp.text}")


async def handle_ingest(args: argparse.Namespace) -> None:
    """Ingest a directory of notes into local SQLite vector store."""
    settings = get_settings()
    folder_path = Path(args.folder).resolve()

    print(f"Starting ingestion for directory: {folder_path}")
    memory_db = MemoryDB(db_path=settings.db_path, encryption_key=settings.memory_encryption_key)
    vector_store = SQLiteVectorStore(memory_db=memory_db)
    embedder = OllamaEmbedder(base_url=settings.ollama_base_url, model_name=settings.embed_model)

    pipeline = DocumentIngestionPipeline(
        memory_db=memory_db,
        vector_store=vector_store,
        embedder=embedder,
    )

    try:
        stats = await pipeline.ingest_directory(folder_path)
        print(f"Ingestion complete: {stats['indexed']} indexed, {stats['skipped']} unchanged (total: {stats['total_scanned']})")
    except Exception as e:
        print(f"[Error during ingestion]: {e}", file=sys.stderr)


async def handle_memory(args: argparse.Namespace) -> None:
    """Memory management CLI subcommands."""
    settings = get_settings()
    target_host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    base_url = (getattr(args, "url", None) or f"http://{target_host}:{settings.port}").rstrip("/")
    token = getattr(args, "token", None) or settings.auth_token
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        if args.memory_action == "list":
            params = {}
            if args.category:
                params["category"] = args.category
            resp = await client.get(f"{base_url}/memory/facts", headers=headers, params=params)
            if resp.status_code == 200:
                facts = resp.json().get("facts", [])
                print(f"--- Stored Facts ({len(facts)}) ---")
                for f in facts:
                    curr = " [CURRENT]" if f.get("is_current") else " [SUPERSEDED]"
                    print(f"- [{f['category'].upper()}]{curr} {f['text']} (id: {f['id'][:8]}...)")
            else:
                print(f"Failed to list facts: {resp.status_code} {resp.text}")

        elif args.memory_action == "forget":
            payload = {"query": args.query, "confirm": args.confirm}
            resp = await client.post(f"{base_url}/memory/forget", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                if data["confirmed"]:
                    print(f"Deleted {data['deleted_count']} memories matching '{data['query']}'.")
                else:
                    print(f"Found {len(data['matches'])} memory match(es) for '{data['query']}':")
                    for m in data["matches"]:
                        print(f"  * [{m['entity_type']}] (sim: {m['similarity']}): {m['preview']}")
                    print("\nRun again with '--confirm' to delete these memories permanently.")
            else:
                print(f"Failed to forget: {resp.status_code} {resp.text}")

        elif args.memory_action == "export":
            resp = await client.get(f"{base_url}/memory/export", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if args.output:
                    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
                    print(f"Memory exported successfully to {args.output}")
                else:
                    print(json.dumps(data, indent=2))
            else:
                print(f"Failed to export: {resp.status_code} {resp.text}")


def main():
    parser = argparse.ArgumentParser(description="Nino AI Assistant CLI")
    parser.add_argument("--url", help="Base URL of Nino gateway (default: http://127.0.0.1:8000)")
    parser.add_argument("--token", help="Bearer authentication token")
    parser.add_argument("--model", help="Override target model name")
    parser.add_argument("--session", help="Specific session ID to load/resume")
    parser.add_argument("--new", action="store_true", help="Start a new session instead of resuming previous")
    parser.add_argument("--no-resume", action="store_true", help="Do not auto-resume previous session")
    parser.add_argument("--speak", action="store_true", help="Speak assistant responses aloud using Piper neural TTS")
    parser.add_argument("--tts-voice", "--voice-model", dest="tts_voice", help="Specify Piper TTS voice name or alias (e.g. kristin, cori, amy, lessac, ryan, alan)")

    subparsers = parser.add_subparsers(dest="subcommand")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a directory of notes for RAG")
    ingest_parser.add_argument("folder", help="Path to folder containing .md, .txt, or .pdf files")

    # Memory subcommand
    memory_parser = subparsers.add_parser("memory", help="Memory management operations")
    mem_subparsers = memory_parser.add_subparsers(dest="memory_action", required=True)

    mem_list = mem_subparsers.add_parser("list", help="List stored personal facts")
    mem_list.add_argument("--category", help="Filter by fact category")

    mem_forget = mem_subparsers.add_parser("forget", help="Search and forget memories")
    mem_forget.add_argument("query", help="Text to search and forget")
    mem_forget.add_argument("--confirm", action="store_true", help="Confirm permanent deletion")

    mem_export = mem_subparsers.add_parser("export", help="Export memory dump to JSON")
    mem_export.add_argument("--output", help="Optional output JSON file path")

    # Confirmations subcommand
    conf_parser = subparsers.add_parser("confirmations", help="Interactive tool confirmations")
    conf_subparsers = conf_parser.add_subparsers(dest="conf_action", required=True)

    conf_subparsers.add_parser("list", help="List pending confirmations")

    conf_approve = conf_subparsers.add_parser("approve", help="Approve a confirmation token")
    conf_approve.add_argument("id", help="Confirmation ID to approve")

    conf_deny = conf_subparsers.add_parser("deny", help="Deny a confirmation token")
    conf_deny.add_argument("id", help="Confirmation ID to deny")

    args = parser.parse_args()

    if args.subcommand == "ingest":
        asyncio.run(handle_ingest(args))
    elif args.subcommand == "memory":
        asyncio.run(handle_memory(args))
    elif args.subcommand == "confirmations":
        asyncio.run(handle_confirmations(args))
    else:
        asyncio.run(chat_interactive(args))


if __name__ == "__main__":
    main()

