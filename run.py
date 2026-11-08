"""Nino Personal AI Launcher.

Easy launcher script for starting Nino backend, interactive CLI, or voice client.
Automatically uses the virtual environment (.venv) if present.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser


WORKSPACE_DIR = Path(__file__).parent.resolve()
VENV_PYTHON_WIN = WORKSPACE_DIR / ".venv" / "Scripts" / "python.exe"
VENV_PYTHON_UNIX = WORKSPACE_DIR / ".venv" / "bin" / "python"


def get_network_ips() -> list[tuple[str, str]]:
    """Detect local network IPv4 addresses (Wi-Fi/LAN)."""
    import socket
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
        if lan_ip and not lan_ip.startswith("127."):
            ips.append(("Wi-Fi / Local Network", lan_ip))
    except Exception:
        pass

    return ips


def print_server_access_urls(port: int):
    """Print clearly formatted access URLs for localhost and other devices on LAN."""
    print("=" * 70)
    print("  * Nino Web Chat Gateway:")
    print(f"    - This Computer:  http://localhost:{port}/")
    net_ips = get_network_ips()
    if net_ips:
        for label, ip in net_ips:
            print(f"    - Other Devices ({label}): http://{ip}:{port}/")
    else:
        print(f"    - Other Devices: http://<your-local-ip>:{port}/")
    print("=" * 70)


def get_python_exe() -> str:
    """Find the best python executable, favoring .venv."""
    if os.name == "nt" and VENV_PYTHON_WIN.is_file():
        return str(VENV_PYTHON_WIN)
    elif VENV_PYTHON_UNIX.is_file():
        return str(VENV_PYTHON_UNIX)
    return sys.executable


def is_backend_running(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Check whether Nino backend is already responding on host:port."""
    check_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        url = f"http://{check_host}:{port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_server_process(py_exe: str, host: str = "127.0.0.1", port: int = 8000) -> subprocess.Popen:
    """Start Nino FastAPI backend server in a background process."""
    cmd = [
        py_exe,
        "-m", "uvicorn",
        "nino.main:app",
        "--host", host,
        "--port", str(port),
        "--no-access-log",
        "--log-level", "warning",
    ]
    print(f"[*] Starting Nino Backend API Server on port {port} (binding {host}) ...")
    print_server_access_urls(port)
    logs_dir = WORKSPACE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    server_log = open(logs_dir / "server.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKSPACE_DIR),
        stdout=server_log,
        stderr=server_log,
    )
    proc._log_file = server_log
    return proc


def wait_for_server(host: str = "127.0.0.1", port: int = 8000, timeout_sec: float = 12.0) -> bool:
    """Wait for backend health check to return OK."""
    start = time.time()
    while time.time() - start < timeout_sec:
        if is_backend_running(host, port):
            return True
        time.sleep(0.5)
    return False


def run_chat(py_exe: str, extra_args: list[str]) -> int:
    """Run interactive streaming chat CLI."""
    cmd = [py_exe, "-m", "nino.cli"] + extra_args
    return subprocess.call(cmd, cwd=str(WORKSPACE_DIR))


def run_voice(py_exe: str, extra_args: list[str]) -> int:
    """Run voice client with push-to-talk."""
    cmd = [py_exe, "-m", "nino.voice.client", "--ptt"] + extra_args
    try:
        return subprocess.call(cmd, cwd=str(WORKSPACE_DIR))
    except KeyboardInterrupt:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Nino Local-First Personal AI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run.py             # Start backend (if needed) & launch interactive chat with persistent memory
  python run.py --ui        # Launch backend and open the Web Chat Interface in browser
  python run.py --server    # Run only the FastAPI backend server
  python run.py --chat      # Run only the interactive CLI chat
  python run.py --voice     # Run Push-To-Talk voice interface
  python run.py --new       # Start a fresh conversation session without resuming previous
""",
    )
    parser.add_argument("--server", action="store_true", help="Start only the FastAPI backend server")
    parser.add_argument("--ui", "--web", dest="web_ui", action="store_true", help="Launch backend & open Web Chat Interface in browser")
    parser.add_argument("--chat", action="store_true", help="Launch interactive CLI chat")
    parser.add_argument("--voice", action="store_true", help="Launch real-time Push-to-Talk voice client")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"), help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--new", action="store_true", help="Start a new session without resuming previous")
    parser.add_argument("--session", help="Specific session ID to resume")
    parser.add_argument("--speak", action="store_true", help="Speak assistant replies aloud in CLI chat using Piper TTS")
    parser.add_argument("--tts-voice", dest="tts_voice", help="Specify Piper TTS voice name or alias (e.g. kristin, cori, amy, lessac, ryan, alan)")

    # Support shorthand positional commands like: python run.py chat / python run.py server
    sub_cmd = sys.argv[1].lower() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    parsed, unknown = parser.parse_known_args()

    py_exe = get_python_exe()
    local_browser_host = "localhost" if parsed.host in ("0.0.0.0", "::") else parsed.host
    local_browser_url = f"http://{local_browser_host}:{parsed.port}/"

    # 1. Server-only mode
    if parsed.server or sub_cmd == "server":
        print_server_access_urls(parsed.port)
        cmd = [py_exe, "-m", "uvicorn", "nino.main:app", "--host", parsed.host, "--port", str(parsed.port), "--reload"]
        sys.exit(subprocess.call(cmd, cwd=str(WORKSPACE_DIR)))

    # 2. Web UI mode
    if parsed.web_ui or sub_cmd in ("ui", "web"):
        print_server_access_urls(parsed.port)
        if not is_backend_running(parsed.host, parsed.port):
            print(f"[*] Starting Nino Backend Server...")
            print(f"[*] Opening Nino Web Chat in your browser: {local_browser_url}")
            try:
                webbrowser.open(local_browser_url)
            except Exception:
                pass
            cmd = [py_exe, "-m", "uvicorn", "nino.main:app", "--host", parsed.host, "--port", str(parsed.port), "--reload"]
            sys.exit(subprocess.call(cmd, cwd=str(WORKSPACE_DIR)))
        else:
            print(f"[+] Nino Backend is already running on port {parsed.port}")
            print(f"[*] Opening Nino Web Chat in your browser: {local_browser_url}")
            try:
                webbrowser.open(local_browser_url)
            except Exception:
                pass
            sys.exit(0)

    # 3. Voice client mode
    if parsed.voice or sub_cmd == "voice":
        srv_proc = None
        if not is_backend_running(parsed.host, parsed.port):
            print(f"[!] Backend server is not running on {parsed.host}:{parsed.port}.")
            print(f"[*] Starting backend server first...")
            srv_proc = start_server_process(py_exe, parsed.host, parsed.port)
            if not wait_for_server(parsed.host, parsed.port):
                print("[-] Failed to connect to server after startup.")
                if srv_proc:
                    srv_proc.terminate()
                sys.exit(1)
        try:
            sys.exit(run_voice(py_exe, unknown))
        finally:
            if srv_proc:
                print("\n[*] Stopping background backend server...")
                srv_proc.terminate()
                if hasattr(srv_proc, "_log_file") and srv_proc._log_file:
                    try:
                        srv_proc._log_file.close()
                    except Exception:
                        pass

    # 4. Interactive Chat mode (default behavior)
    # Check if server is running; if not, spin it up
    srv_proc = None
    if not is_backend_running(parsed.host, parsed.port):
        print(f"[*] Backend server not detected on port {parsed.port}.")
        srv_proc = start_server_process(py_exe, parsed.host, parsed.port)
        if not wait_for_server(parsed.host, parsed.port):
            print("[-] Timed out waiting for backend server to start.")
            if srv_proc:
                srv_proc.terminate()
            sys.exit(1)
        print("[+] Backend server is ready!\n")
    else:
        print(f"[+] Connected to existing backend server on port {parsed.port}\n")
        print_server_access_urls(parsed.port)

    chat_args = list(unknown)
    if parsed.new:
        chat_args.append("--new")
    if parsed.session:
        chat_args.extend(["--session", parsed.session])
    if parsed.speak:
        chat_args.append("--speak")
    if parsed.tts_voice:
        chat_args.extend(["--tts-voice", parsed.tts_voice])

    try:
        run_chat(py_exe, chat_args)
    finally:
        # If we started a temporary background server process in this session, terminate it on exit
        if srv_proc:
            print("\n[*] Stopping background backend server...")
            srv_proc.terminate()
            if hasattr(srv_proc, "_log_file") and srv_proc._log_file:
                try:
                    srv_proc._log_file.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
