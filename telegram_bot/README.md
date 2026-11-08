# Nino AI Assistant - Secure Telegram Bot Interface

A private, secure, local-first Telegram interface for **Nino**, powered by your local **Ollama API** (e.g. `gemma2:2b`), **python-telegram-bot** (v20+), **faster-whisper** (STT), and **edge-tts** (TTS).

---

## 🔒 Key Security & Privacy Guarantees

1. **Strict User Gatekeeper**:
   - Nino communicates **only with you**.
   - Every incoming message and command is gated against `MY_TELEGRAM_USER_ID`.
   - Any message from any other Telegram user is logged as an unauthorized intrusion and silently discarded.

2. **Long Polling Architecture (`run_polling`)**:
   - Zero public inbound open ports.
   - Zero webhook routing or external domain certificates needed.
   - Safe to run behind local firewalls, VPNs, or home networks.

3. **100% Local Inference via Ollama**:
   - Nino communicates with your local Ollama instance (`http://127.0.0.1:11434`).
   - Zero cloud model API keys required. Your conversations stay strictly on-device.

4. **Antigravity Security Policies**:
   - `deny("command(*)")`: Prevents arbitrary operating system shell or command execution.
   - `deny("read_file(*)")`: Blocks unrestricted disk inspection.
   - `deny("execute_url(*)")`: Disallows unmanaged outbound network calls.
   - **Gated Vault Tool**: Access to local notes (`./vault`) requires explicit interactive confirmation in Telegram.

---

## 📁 Project Structure

```
telegram_bot/
├── bot.py             # Main Telegram polling runner & security filter
├── agent.py           # Ollama Agent definition & Antigravity security policies
├── voice_utils.py     # Async faster-whisper STT & edge-tts voice helpers
├── requirements.txt   # Python package dependencies
├── .env.example       # Template for environment variables
└── README.md          # Documentation & setup guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Credentials

1. **Ollama Running Locally**:
   - Ensure Ollama is installed and running (`ollama serve`).
   - Pull your preferred model (default: `gemma2:2b`):
     ```sh
     ollama pull gemma2:2b
     ```

2. **Telegram Bot Token**:
   - Message [@BotFather](https://t.me/botfather) on Telegram.
   - Send `/newbot`, name your bot (e.g. `MyNinoBot`), and copy the HTTP API Token.

3. **Your Telegram User ID**:
   - Message [@userinfobot](https://t.me/userinfobot) or [@rawdatabot](https://t.me/rawdatabot) on Telegram.
   - Copy your numeric `Id` (e.g. `1234567890`).

cp .env.example .env
```

Edit `.env` and fill in your credentials:

```ini
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxyz
MY_TELEGRAM_USER_ID=1234567890
GEMINI_API_KEY=AIzaSy...
```

### 3. Install Dependencies

In your Python environment:

```sh
pip install -r requirements.txt
```

### 4. Run the Bot

```sh
python bot.py
```

You will see:
```text
==================================================================
  Nino AI Assistant • Secure Telegram Bot Interface
  Authorized User ID: 1234567890
  Antigravity Policies: deny('command(*)'), deny('read_file(*)'), deny('execute_url(*)')
  STT Model: faster-whisper (base.en) | TTS Voice: en-US-JennyNeural
  Running with Long Polling (zero inbound ports / webhooks needed)
==================================================================
```

Open Telegram, find your bot, and send `/start`.

---

## 💬 Bot Features & Commands

* **Text Conversations**: Chat naturally. Nino speaks in her signature cool, witty, and loyal persona.
* **Inbound Voice Notes**: Record and send voice notes. Nino transcribes them via `faster-whisper`, reasons, and replies with audio.
* **Outbound Voice Notes**: Nino vocalizes responses using `edge-tts` and sends them back as native Telegram audio notes.
* `/mode <companion|work|focus>`: Switch between casual witty companion, task-focused work mode, or minimalist deep-focus mode.
* `/vault <filename>`: Request access to a secure note in `./vault/`. Nino will ask for interactive approval via inline buttons.
* `/voice [on|off]`: Toggle audio voice replies on or off (disabled by default; replies in text-only mode until toggled on).
* **Smart Sticker Replies**: Send any sticker to Nino; she visually and emotionally reads the sticker and accurately replies with a matching sticker from the same pack or emotional sentiment.
* `/clear`: Clear conversation memory.
