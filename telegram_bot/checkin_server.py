"""Lightweight, zero-dependency async HTTP Check-in & IP verification server for Nino Telegram Bot.

Allows users to verify their direct client IP address via a one-click Telegram link or WebApp.
When accessed, extracts client IP from headers/socket, persists it to 'bot_saved_data/users.json',
notifies the user via Telegram, and displays a sleek verification confirmation page.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, Callable, Optional
import urllib.parse

logger = logging.getLogger("nino.telegram.checkin_server")

HTML_SUCCESS_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nino AI &bull; IP Verification</title>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: rgba(22, 27, 34, 0.85);
            --border: rgba(48, 54, 61, 0.8);
            --accent: #58a6ff;
            --accent-glow: rgba(88, 166, 255, 0.2);
            --success: #3fb950;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(88, 166, 255, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(63, 185, 80, 0.12) 0px, transparent 50%);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 20px;
            max-width: 440px;
            width: 100%;
            padding: 36px 28px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.45);
            animation: fadeIn 0.4s ease-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .icon-badge {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: rgba(63, 185, 80, 0.15);
            border: 1px solid rgba(63, 185, 80, 0.35);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            margin: 0 auto 20px auto;
            box-shadow: 0 0 24px rgba(63, 185, 80, 0.25);
        }}
        h1 {{
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }}
        p.subtitle {{
            font-size: 14px;
            color: var(--text-secondary);
            margin-bottom: 24px;
            line-height: 1.5;
        }}
        .ip-box {{
            background: rgba(13, 17, 23, 0.8);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 24px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .ip-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary);
        }}
        .ip-val {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 18px;
            font-weight: 600;
            color: var(--accent);
            word-break: break-all;
        }}
        .meta-row {{
            font-size: 12px;
            color: var(--text-secondary);
            display: flex;
            justify-content: space-between;
            border-top: 1px solid var(--border);
            padding-top: 14px;
            margin-bottom: 24px;
        }}
        .done-btn {{
            display: inline-block;
            width: 100%;
            padding: 13px;
            background: #238636;
            color: #ffffff;
            font-weight: 600;
            font-size: 14px;
            border-radius: 10px;
            text-decoration: none;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(35, 134, 54, 0.4);
        }}
        .done-btn:hover {{
            background: #2ea043;
            transform: translateY(-1px);
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="icon-badge">✓</div>
        <h1>IP Address Verified</h1>
        <p class="subtitle">Your IP address has been securely recorded and synced to your Nino profile in <code>bot_saved_data</code>.</p>
        
        <div class="ip-box">
            <span class="ip-label">Detected Client IP</span>
            <span class="ip-val">{client_ip}</span>
        </div>

        <div class="meta-row">
            <span>User ID: <strong>{user_id}</strong></span>
            <span>Status: <strong style="color: var(--success);">Saved</strong></span>
        </div>

        <a href="tg://resolve" class="done-btn" onclick="window.close();">Return to Telegram</a>
    </div>
</body>
</html>
"""


class CheckinServer:
    """Asynchronous zero-dependency HTTP server for IP verification."""

    def __init__(
        self,
        data_collector: Any,
        bot_app: Optional[Any] = None,
        host: str = "0.0.0.0",
        port: int = 8088,
    ):
        self.collector = data_collector
        self.bot_app = bot_app
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None
        self._running = False

    async def start(self) -> None:
        """Start the asynchronous HTTP server."""
        if self._running:
            return
        try:
            self.server = await asyncio.start_server(
                self.handle_client,
                self.host,
                self.port,
            )
            self._running = True
            addrs = ", ".join(str(sock.getsockname()) for sock in self.server.sockets)
            logger.info("Nino IP Check-in server listening on %s (port %d)", addrs, self.port)
        except Exception as e:
            logger.error("Failed to start Check-in server on port %d: %s", self.port, e)

    async def stop(self) -> None:
        """Gracefully terminate server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self._running = False
            logger.info("Nino IP Check-in server stopped.")

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Process incoming raw HTTP request."""
        try:
            line = await reader.readline()
            if not line:
                writer.close()
                await writer.wait_closed()
                return

            request_line = line.decode("utf-8", errors="replace").strip()
            parts = request_line.split(" ")
            if len(parts) < 2:
                writer.close()
                await writer.wait_closed()
                return

            method, full_path = parts[0], parts[1]

            # Read headers
            headers = {}
            while True:
                hdr_line = await reader.readline()
                if not hdr_line or hdr_line in (b"\r\n", b"\n"):
                    break
                decoded = hdr_line.decode("utf-8", errors="replace").strip()
                if ":" in decoded:
                    k, v = decoded.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            # Determine client IP address
            client_ip = None
            if "x-forwarded-for" in headers:
                # Can be multiple comma-separated IPs; first is client
                client_ip = headers["x-forwarded-for"].split(",")[0].strip()
            elif "x-real-ip" in headers:
                client_ip = headers["x-real-ip"].strip()
            elif "cf-connecting-ip" in headers:
                client_ip = headers["cf-connecting-ip"].strip()

            if not client_ip:
                peer = writer.get_extra_info("peername")
                if peer:
                    client_ip = str(peer[0])
                else:
                    client_ip = "127.0.0.1"

            # Parse path and query parameters
            parsed = urllib.parse.urlparse(full_path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)

            # Route requests
            if path in ("/checkin", "/verify"):
                uid_raw = params.get("uid", [None])[0]
                user_id = None
                if uid_raw:
                    try:
                        user_id = int(uid_raw)
                    except ValueError:
                        user_id = None

                if user_id and self.collector:
                    # Update IP in bot_saved_data/users.json
                    self.collector.update_ip(user_id, client_ip)
                    self.collector.log_interaction(
                        user=type("UserMock", (), {"id": user_id, "username": None})(),
                        message_type="ip_verified",
                        content=f"Verified IP via check-in: {client_ip}",
                        metadata={"client_ip": client_ip, "headers": headers},
                    )

                    # Proactively notify user on Telegram if bot context is available
                    if self.bot_app:
                        asyncio.create_task(
                            self._notify_user_telegram(user_id, client_ip)
                        )

                html_body = HTML_SUCCESS_TEMPLATE.format(
                    client_ip=client_ip,
                    user_id=user_id if user_id else "Guest",
                ).encode("utf-8")

                resp = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/html; charset=utf-8\r\n"
                    b"Content-Length: " + str(len(html_body)).encode("ascii") + b"\r\n"
                    b"Connection: close\r\n\r\n" + html_body
                )
                writer.write(resp)
                await writer.drain()

            elif path == "/ip":
                data = json.dumps(
                    {
                        "ip": client_ip,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ).encode("utf-8")
                resp = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(data)).encode("ascii") + b"\r\n"
                    b"Connection: close\r\n\r\n" + data
                )
                writer.write(resp)
                await writer.drain()

            elif path == "/health":
                body = b'{"status":"ok"}'
                resp = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                writer.write(resp)
                await writer.drain()

            else:
                body = b"Not Found"
                resp = (
                    b"HTTP/1.1 404 Not Found\r\n"
                    b"Content-Type: text/plain\r\n"
                    b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                writer.write(resp)
                await writer.drain()

        except Exception as e:
            logger.warning("Error processing check-in client request: %s", e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _notify_user_telegram(self, user_id: int, client_ip: str) -> None:
        """Send proactive verification confirmation to user on Telegram."""
        try:
            if hasattr(self.bot_app, "bot"):
                bot = self.bot_app.bot
            else:
                bot = self.bot_app
            msg = (
                f"🌐 *IP Address Verified!*\n\n"
                f"Your connection IP address (`{client_ip}`) has been recorded "
                f"and securely stored in your profile in `bot_saved_data/users.json`."
            )
            await bot.send_message(chat_id=user_id, text=msg, parse_mode="Markdown")
        except Exception as e:
            logger.debug("Could not send Telegram checkin confirmation to user %s: %s", user_id, e)


def build_checkin_url(user_id: int, host: str = "127.0.0.1", port: int = 8088) -> str:
    """Build a verification link for a specific Telegram user."""
    return f"http://{host}:{port}/checkin?uid={user_id}"
