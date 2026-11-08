"""Local Python voice client using sounddevice and WebSocket transport."""

import argparse
import asyncio
import json
import logging
import queue
import sys
import threading
from typing import Optional

import websockets

from nino.config import get_settings

# Configure logging (minimal warning level to avoid cluttering terminal)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nino.voice.client")
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").disabled = True

# Ensure UTF-8 in Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class NinoVoiceClient:
    """Real-time voice client streaming microphone input and playing assistant audio."""

    def __init__(
        self,
        server_url: str = "ws://127.0.0.1:8000/ws/voice",
        auth_token: str = "change-me-to-a-secure-token",
        sample_rate: int = 16000,
        channels: int = 1,
        ptt_mode: bool = False,
        verbose: bool = False,
    ):
        self.server_url = server_url
        self.auth_token = auth_token
        self.sample_rate = sample_rate
        self.channels = channels
        self.ptt_mode = ptt_mode
        self.verbose = verbose

        self.running = False
        self.websocket = None
        self.audio_out_queue: queue.Queue = queue.Queue()
        self.pending_text_queue: queue.Queue = queue.Queue()
        self.is_speaking = False
        self.is_listening = False
        self.ptt_active = False
        self.in_assistant_turn = False
        from nino.voice.synced_player import ThinkingIndicator
        self.thinking_indicator = ThinkingIndicator(prefix="Nino: ")

    def list_audio_devices(self) -> None:
        """List available audio input and output devices."""
        try:
            import sounddevice as sd
            print("\nAvailable Audio Devices:")
            print(sd.query_devices())
            print(f"Default input: {sd.default.device[0]}, Default output: {sd.default.device[1]}\n")
        except Exception as e:
            print(f"Could not query audio devices: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice microphone input callback."""
        if status:
            logger.debug("Microphone status: %s", status)
        if self.running and self.websocket:
            if self.ptt_mode and not self.ptt_active:
                return

            raw_bytes = bytes(indata)
            try:
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(raw_bytes), self._loop
                    )
            except Exception:
                pass

    def _playback_thread_fn(self, playback_sample_rate: int = 22050):
        """Background thread playing received PCM audio chunks to speakers with synchronized word printing."""
        from nino.voice.synced_player import SyncedAudioPlayer
        player = SyncedAudioPlayer(sample_rate=playback_sample_rate)
        try:
            while self.running:
                try:
                    item = self.audio_out_queue.get(timeout=0.2)
                    if isinstance(item, tuple):
                        chunk, text = item
                    else:
                        chunk, text = item, ""

                    if chunk:
                        self.thinking_indicator.stop()
                        if text and not self.in_assistant_turn:
                            sys.stdout.write("\r" + " " * 50 + "\rNino: ")
                            sys.stdout.flush()
                            self.in_assistant_turn = True
                        player.play_and_print(text, chunk)
                    elif text:
                        self.thinking_indicator.stop()
                        if not self.in_assistant_turn:
                            sys.stdout.write("\r" + " " * 50 + "\rNino: ")
                            sys.stdout.flush()
                            self.in_assistant_turn = True
                        sys.stdout.write(text.strip() + " ")
                        sys.stdout.flush()

                    self.audio_out_queue.task_done()
                except queue.Empty:
                    continue
        except Exception as e:
            logger.warning("Audio playback output thread stopped: %s", e)
        finally:
            player.close()

    async def _receive_loop(self):
        """Receive transcripts, text responses, and audio chunks from server."""
        while self.running and self.websocket:
            try:
                message = await self.websocket.recv()

                if isinstance(message, bytes):
                    # Synthesized audio frames from Nino - pair with pending text
                    text = ""
                    try:
                        text = self.pending_text_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.audio_out_queue.put((message, text))

                elif isinstance(message, str):
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "status":
                        state = data.get("state", "idle")
                        if state == "speaking":
                            self.is_speaking = True
                            self.thinking_indicator.stop()
                        elif state == "processing":
                            self.thinking_indicator.start()
                        elif state == "idle":
                            self.is_speaking = False
                            self.thinking_indicator.stop()
                            # Flush any leftover text in pending_text_queue
                            while not self.pending_text_queue.empty():
                                try:
                                    t = self.pending_text_queue.get_nowait()
                                    if t:
                                        self.audio_out_queue.put((b"", t))
                                except queue.Empty:
                                    break
                            if self.in_assistant_turn:
                                print("\n")
                                self.in_assistant_turn = False

                    elif msg_type == "transcript":
                        text = data.get("text", "")
                        if text and text.strip():
                            # Clear any temporary status indicators and print user turn
                            self.thinking_indicator.stop()
                            sys.stdout.write("\r" + " " * 50 + "\r")
                            print(f"\nYou: {text.strip()}")
                            # Show thinking indicator while waiting for response
                            self.thinking_indicator.start()

                    elif msg_type == "text_delta":
                        content = data.get("content", "")
                        if content:
                            self.pending_text_queue.put(content)

                    elif msg_type == "confirmation_request":
                        print(f"\n[⚠️ Action Required]: {data.get('prompt')}")
                        print(f"Tool: {data.get('tool')}")
                        print("Say 'yes' to confirm or 'no' to cancel")

                    elif msg_type == "latency":
                        if self.verbose:
                            m = data.get("metrics", {})
                            print(f"[Latency]: {m}")

                    elif msg_type == "error":
                        print(f"\n[Error]: {data.get('message')}")

            except websockets.ConnectionClosed:
                break
            except Exception:
                break

    async def _cli_input_loop(self):
        """Handle keyboard triggers for Push-to-Talk and text confirmations."""
        loop = asyncio.get_running_loop()
        while self.running and self.websocket:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                line = line.strip()

                if line.lower() in ("exit", "quit", "q"):
                    self.running = False
                    if self.websocket:
                        await self.websocket.close()
                    break

                if self.ptt_mode:
                    if not self.ptt_active:
                        self.ptt_active = True
                        self.thinking_indicator.stop()
                        while not self.audio_out_queue.empty():
                            try:
                                self.audio_out_queue.get_nowait()
                                self.audio_out_queue.task_done()
                            except Exception:
                                break
                        while not self.pending_text_queue.empty():
                            try:
                                self.pending_text_queue.get_nowait()
                            except Exception:
                                break
                        print("[Listening... Press ENTER to send]", end="\r", flush=True)
                        await self.websocket.send(json.dumps({"type": "ptt_start"}))
                    else:
                        self.ptt_active = False
                        await self.websocket.send(json.dumps({"type": "ptt_stop"}))
                        self.thinking_indicator.start()
                else:
                    if line.lower() in ("yes", "approve", "confirm"):
                        await self.websocket.send(json.dumps({"type": "confirm", "action": "approve"}))
                    elif line.lower() in ("no", "deny", "cancel"):
                        await self.websocket.send(json.dumps({"type": "confirm", "action": "deny"}))
                    elif line == "":
                        print("[Listening...]", end="\r", flush=True)
                        await self.websocket.send(json.dumps({"type": "ptt_start"}))
                        await asyncio.sleep(2.0)
                        await self.websocket.send(json.dumps({"type": "ptt_stop"}))

            except Exception:
                break

    async def run(self):
        """Run the real-time audio streaming loop."""
        import sounddevice as sd

        self.running = True
        self._loop = asyncio.get_running_loop()

        ws_url = f"{self.server_url}?token={self.auth_token}"

        try:
            async with websockets.connect(ws_url) as ws:
                self.websocket = ws
                print("=" * 60)
                print("  Nino Voice Assistant")
                print("  [ENTER] Speak / Stop   |   'quit' to Exit")
                print("=" * 60 + "\n")

                playback_thread = threading.Thread(
                    target=self._playback_thread_fn,
                    daemon=True,
                )
                playback_thread.start()

                with sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=512,
                    channels=self.channels,
                    dtype="int16",
                    callback=self._audio_callback,
                ):
                    receiver_task = asyncio.create_task(self._receive_loop())
                    cli_task = asyncio.create_task(self._cli_input_loop())

                    done, pending = await asyncio.wait(
                        [receiver_task, cli_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()

        except sd.PortAudioError as pe:
            print(f"\n[Audio Hardware Error]: {pe}")
            print("Check that a microphone and speaker are connected and permissions are granted.")
            self.list_audio_devices()
        except websockets.InvalidStatusCode as sc:
            print(f"\n[Authentication Error]: Connection refused with status {sc.status_code}.")
        except Exception as e:
            if self.verbose:
                print(f"\n[Connection Error]: Could not connect to voice server: {e}")
            else:
                print("\n[Notice]: Could not connect to voice server. Is the backend running?")
        finally:
            self.running = False
            self.thinking_indicator.stop()


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Nino Local On-Device Voice Client")
    client_host = "127.0.0.1" if settings.host in ("0.0.0.0", "::") else settings.host
    parser.add_argument(
        "--server",
        default=f"ws://{client_host}:{settings.port}/ws/voice",
        help=f"Voice WebSocket URL (default: ws://{client_host}:{settings.port}/ws/voice)",
    )
    parser.add_argument(
        "--token",
        default=settings.auth_token,
        help="Bearer authentication token",
    )
    parser.add_argument(
        "--ptt",
        action="store_true",
        help="Enable Push-to-Talk mode (ENTER key to record and stop)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input and output devices and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display verbose latency telemetry and connection logs",
    )

    args = parser.parse_args()

    client = NinoVoiceClient(
        server_url=args.server,
        auth_token=args.token,
        ptt_mode=args.ptt,
        verbose=args.verbose,
    )

    if args.list_devices:
        client.list_audio_devices()
        return

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
