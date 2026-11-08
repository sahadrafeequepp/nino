# Nino Voice Subsystem Testing & Verification Guide

This document outlines the manual verification checklist, hardware requirements, latency profiling procedures, and the roadmap for training a custom "Hey Nino" wake word model.

---

## 1. Automated Test Suite Summary

The automated unit and integration tests run entirely offline with synthetic PCM audio and mocked models:
- **`tests/test_voice_streaming.py`**: Validates punctuation-aware sentence boundary splitting (protecting abbreviations like `Dr.` / `Mr.` and decimals like `3.14`), and verifies that sentence 1 is emitted for synthesis before later LLM tokens finish.
- **`tests/test_voice_confirmation.py`**: Tests spoken confirmation safety. Confirms that only clear affirmative phrases (`yes`, `confirm`, `approve`) approve sensitive tools, explicit negatives (`no`, `cancel`) cancel, and ambiguous/mumbled answers (`maybe`, `uhh`, `what?`) are strictly rejected.
- **`tests/test_voice_duplex.py`**: Tests half-duplex microphone muting while speaking, and experimental barge-in interruption logic.
- **`tests/test_voice_telemetry.py`**: Tests latency checkpoint recording ($T_{\text{speech\_end}} \to T_{\text{stt}} \to T_{\text{first\_token}} \to T_{\text{first\_audio}}$) and math.
- **`tests/test_voice_privacy.py`**: Enforces Local-Only Media Quarantine (refusal to route audio bytes or data URIs through Privacy Gateway) and confirms zero audio files written to disk (`STORE_AUDIO=false`).
- **`tests/test_voice_ws.py`**: Tests the `/ws/voice` WebSocket endpoint authentication and round-trip turn exchange.

Run all automated voice tests:
```powershell
.venv\Scripts\pytest -k voice -v
```

---

## 2. Manual Verification Checklist (Live Audio & Microphone)

These checks require a physical microphone and speaker setup.

### Test 1: Device Enumeration
Verify that `sounddevice` detects your input (microphone) and output (speakers/headphones):
```powershell
.venv\Scripts\python -m app.voice.client --list-devices
```
- [ ] Default input device is detected with $\ge 1$ input channel.
- [ ] Default output device is detected with $\ge 1$ output channel.

### Test 2: Push-to-Talk (PTT) Reliable Fallback
Run the voice client in Push-to-Talk mode:
```powershell
.venv\Scripts\python -m app.voice.client --ptt
```
1. Press `ENTER` to start speaking. Notice the terminal displays `[🔴 RECORDING - Press ENTER to stop]`.
2. Say: *"What time is it?"*
3. Press `ENTER` to stop recording.
4. Verify:
   - [ ] Transcript appears: `[🗣️ You]: What time is it?`
   - [ ] Nino streams tokens to terminal and plays synthesized audio through speakers.
   - [ ] Latency telemetry is printed: `⏱️ [Latency]: Total: X.XXs (STT: ... | LLM: ... | Audio: ...)`.
   - [ ] Microphone returns to muted/idle state after playback.

### Test 3: Wake Word Activation (Stock: "Hey Jarvis")
Run the voice client in continuous wake-word mode:
```powershell
.venv\Scripts\python -m app.voice.client
```
1. Speak the stock wake word clearly: *"Hey Jarvis"*.
2. Verify:
   - [ ] Terminal transitions from `[💤 IDLE]` to `[🎤 MIC ACTIVE / LISTENING] Speak now...`.
3. Ask: *"What is on my task list?"*
4. Stop speaking and pause for $\sim 600$ms.
5. Verify:
   - [ ] VAD detects end of speech automatically and switches to `[⏳ THINKING]`.
   - [ ] Nino responds aloud with speech and text.

### Test 4: Ambient Noise & Threshold Tuning
1. In a noisy room (e.g., background fan or keyboard typing), observe whether false wake word activations occur.
2. If false activations occur, increase the threshold in `.env`:
   ```dotenv
   WAKE_WORD_THRESHOLD=0.65
   VAD_THRESHOLD=0.6
   ```
3. Verify that normal speech triggers wake-up while continuous noise is ignored.

### Test 5: Sensitive Tool Spoken Confirmation (Safety Verification)
1. Ask Nino to perform a sensitive action: *"Delete task number one."*
2. Verify Nino responds aloud:
   `"Nino needs your approval to execute task delete with task_id 1. Please say yes to confirm, or no to cancel."`
3. **Mumbled / Ambiguous Test**:
   - Reply: *"Uhhh maybe, I don't know."*
   - Verify: Nino does **NOT** execute the tool, replies: `"Confirmation was unclear or mumbled. For your safety, the action was cancelled."`
   - Check `python -m nino.cli confirmations list`: confirmation is marked `denied`.
4. **Affirmative Test**:
   - Repeat the request and clearly say: *"Yes"*.
   - Verify: Nino replies `"Action confirmed. Executing now."` and completes the action.

### Test 6: Half-Duplex Muting vs. Experimental Barge-In
1. **Half-Duplex (Default)**:
   - Ask a question that produces a longer answer (e.g. *"Explain what Nino is."*).
   - While Nino is speaking through speakers, speak or cough.
   - Verify: Assistant audio does NOT echo into the microphone or trigger self-conversation loops.
2. **Barge-In (Experimental)**:
   - Put on headphones.
   - Set in `.env`:
     ```dotenv
     VOICE_DUPLEX_MODE=barge_in
     ```
   - Ask a question and speak over Nino while it answers.
   - Verify: Audio playback stops immediately and transitions back to `[🎤 MIC ACTIVE / LISTENING]`.

---

## 3. Hardware Latency Budget & Targets

The design targets $\le 2.0$ seconds from the end of user speech to the first byte of synthesized audio heard:

| Pipeline Stage | Target Latency | Notes |
| :--- | :--- | :--- |
| **VAD Silence Window** | 400 - 600 ms | Configurable via `VAD_SILENCE_TIMEOUT_MS`. Lower values trigger faster but can cut off pauses. |
| **STT Transcription (faster-whisper)** | 200 - 500 ms | `base.en` with `int8` on modern CPU; $< 150$ms on GPU. |
| **LLM Time-to-First-Token (TTFT)** | 300 - 600 ms | Dependent on Ollama model size (`gemma2:2b` or `qwen2.5:3b`). |
| **Sentence 1 Boundary & Piper TTS** | 100 - 250 ms | Piper synthesizes sentence 1 in parallel while LLM streams sentence 2. |
| **Total Speech-to-Audio** | **$\le 1.8$ - 2.0 s** | Logged automatically per turn. |

---

## 4. Roadmap: Training a Custom "Hey Nino" Wake Word

In future updates, a custom "Hey Nino" wake word model can be trained and integrated using openWakeWord:

### Training Workflow
1. **Synthetic Audio Generation**:
   - Use Piper or multi-speaker TTS to generate 5,000 - 10,000 synthetic audio clips saying *"Hey Nino"* across diverse pitch, rate, and synthetic room impulse response (RIR) reverberations.
2. **Negative Data**:
   - Use openWakeWord's public negative audio dataset ($\ge 2,000$ hours of background speech, conversational podcasts, and room noises).
3. **Feature Extraction**:
   - Feed the clips through openWakeWord's pre-trained audio embedding model (`embedding_model.onnx` and `melspectrogram.onnx`).
4. **Classifier Training**:
   - Train a lightweight 2-layer neural classifier on top of the audio embeddings.
5. **Export to ONNX**:
   - Export the trained model to `hey_nino.onnx` (file size $\sim 1.2$ MB).
6. **Deployment**:
   - Place `hey_nino.onnx` into `data/models/wakewords/`.
   - Update `.env`:
     ```dotenv
     WAKE_WORD_MODEL=data/models/wakewords/hey_nino.onnx
     ```
