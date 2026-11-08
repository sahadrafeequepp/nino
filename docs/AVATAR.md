# Nino Animated Avatar & Persona System (Phase 4.5)

## Overview
Phase 4.5 introduces a rich persona framework, dynamic emotion signaling, and a real-time animated on-screen 3D avatar that coordinates speech playback with lip-sync and visual expressions.

The architecture remains strictly **local-first**, **security-minded**, and adheres to **Rule 6 (Original Character & Asset Policy)** by providing an original 3D procedural avatar out of the box with zero external dependencies, alongside standard VRM 0.0/1.0 loader support for user-supplied models.

---

## 1. Architecture

```
User (Browser / Web Client)
  ├── 3D Viewport (Three.js / @pixiv/three-vrm)
  │    ├── ProceduralAvatar (Default: Original 3D robot/humanoid head)
  │    │    ├── Natural breathing oscillation
  │    │    ├── Dynamic eye blinking & saccades
  │    │    ├── Cursor gaze tracking
  │    │    ├── Emotion-reactive color/lighting
  │    │    └── Speech amplitude mouth scaling
  │    └── VRMAvatar (Optional: User .vrm model with blendshapes)
  ├── Web Audio Player (AudioContext + AnalyserNode)
  │    └── Extracts RMS speech-band amplitude -> drives avatar mouth aperture
  └── Control Panel (Mode switcher, chat drawer, push-to-talk, auth)
            ▲
            │ WebSocket /ws/avatar (token-authenticated)
            ▼
FastAPI Gateway Backend
  ├── PersonaManager (persona/nino.yaml)
  │    ├── Modes: companion, work, focus
  │    ├── Ethical guardrails & prompt composition
  │    └── Crisis detection & safe response
  ├── EmotionParser (nino/persona/emotions.py)
  │    ├── Parses [emotion:<name>] tags across streaming token chunks
  │    ├── Whitelist validation (neutral, happy, sad, angry, surprised, relaxed, thinking, concerned)
  │    └── Strips tags from TTS/chat text
  └── Voice / LLM Pipeline
       ├── SentenceBoundarySplitter & TTS (Kokoro/Piper/Mock)
       └── Streaming audio and token deltas
```

---

## 2. Persona System (`persona/nino.yaml`)

Nino's personality and boundaries are configured in `persona/nino.yaml`:

### Modes
1. **Companion (Default)**: Warm, friendly, supportive, playful, conversational.
2. **Work**: Concise, direct, task-oriented. **Strict boundary**: Absolutely no pet names or affectionate phrasing.
3. **Focus**: Minimalist, terse, distraction-free.

### Switching Modes
Modes can be changed via:
- Web UI mode buttons (`Companion`, `Work`, `Focus`).
- REST endpoint: `POST /persona/mode` with `{"mode": "work"}`.
- WebSocket command: `{"type": "set_mode", "mode": "work"}`.
- Natural language voice/text commands: e.g. *"switch to work mode"*, *"let's go into companion mode"*.

### Ethical Guardrails & Crisis Safety
- **AI Honesty**: Nino always acknowledges it is an AI and never pretends to be human.
- **Non-Possessive**: Never guilt-trips, claims relationship exclusivity, or displays unhealthy attachment.
- **Crisis Protocol**: Any mention of distress or self-harm immediately triggers supportive intervention with official crisis helpline information (e.g. 988 Suicide & Crisis Lifeline) and **strictly bypasses persistent memory storage**.

---

## 3. Emotion Signaling (`[emotion:<name>]`)

The LLM is prompted to prepend emotional reactions to sentences using standard tags:
```
[emotion:happy] Good morning! How can I help you today?
```

### Whitelist
The parser only accepts valid emotions:
- `neutral`
- `happy`
- `sad`
- `angry`
- `surprised`
- `relaxed`
- `thinking`
- `concerned`

### Streaming Chunk Splitting
If an emotion tag is split across LLM streaming tokens (e.g., `"[emo"` in chunk 1, followed by `"tion:happy]"` in chunk 2), `EmotionParser` buffers partial tag segments until complete, emits the emotion event to the WebSocket, and strips the tag from user-facing text and TTS audio.

---

## 4. WebSocket Protocol (`/ws/avatar`)

### Authentication
Connections require a valid gateway token passed either via HTTP Bearer header or query parameter `token`:
```
ws://127.0.0.1:8000/ws/avatar?token=<NINO_AUTH_TOKEN>
```
Unauthorized connections are rejected with code `1008 (WS_1008_POLICY_VIOLATION)`. Tokens are stored solely in client memory / `sessionStorage`.

### Server-to-Client Messages
| Type | Payload | Description |
|---|---|---|
| `state` | `{"type": "state", "state": "idle" \| "listening" \| "thinking" \| "speaking"}` | Conversational state updates |
| `mode` | `{"type": "mode", "mode": "companion"}` | Current active mode |
| `mode_changed` | `{"type": "mode_changed", "mode": "work"}` | Notification when mode changed |
| `emotion` | `{"type": "emotion", "emotion": "happy"}` | Emotion trigger for avatar expression |
| `delta` / `text_delta` | `{"type": "delta", "text": "chunk"}` | Cleaned text stream |
| `text_complete` | `{"type": "text_complete", "emotion": "happy"}` | Turn finalized |
| `audio` | Binary PCM/WAV chunks | Synthesized speech audio |

### Client-to-Server Messages
| Type | Payload | Description |
|---|---|---|
| `user_text` / `chat` | `{"type": "user_text", "text": "hello"}` | User text message |
| `set_mode` | `{"type": "set_mode", "mode": "work"}` | Change persona mode |
| `ptt_start` | `{"type": "ptt_start"}` | Start push-to-talk recording |
| `ptt_stop` | `{"type": "ptt_stop"}` | Stop push-to-talk recording |
| `user_audio` | `{"type": "user_audio", "data": "<base64>"}` | Audio recording upload |

---

## 5. Avatar Rendering & Lip-Sync

### Procedural Avatar (`ProceduralAvatar`)
- Rendered in Three.js with clean, modern cybernetic geometry.
- Natural breathing oscillation along the Y-axis.
- Eye blinking with randomized intervals (~2.5–5s) and natural cosine easing.
- Real-time cursor gaze tracking via head rotation and eye saccades.
- Expressive materials shift color and emissive glow based on emotion (cyan for neutral, emerald for happy, blue for sad, crimson for angry, amber for surprised/thinking, coral for concerned).
- **Rule 6 Compliance**: 100% procedural code, zero external media or copyrighted models.

### Custom VRM Avatar (`VRMAvatar`)
- Powered by `@pixiv/three-vrm` and `three/examples/jsm/loaders/GLTFLoader`.
- Users can drop a `.vrm` file onto the web interface or place one at `assets/avatar/nino.vrm`.
- Maps standard VRM blend shapes:
  - `aa`, `ih`, `ou`, `ee`, `oh` for lip-sync.
  - `blink`, `blinkLeft`, `blinkRight` for eye blinks.
  - `happy`, `sad`, `angry`, `surprised`, `relaxed` for facial expressions.

### Web Audio Analyser Lip-Sync
- Audio chunks played through `AudioContext` are routed through an `AnalyserNode` (FFT size 256).
- Computes speech-band RMS amplitude (100Hz – 3kHz) at 60 FPS.
- Converts amplitude to a normalized aperture `[0.0, 1.0]`, continuously updating `activeRenderer.setLipSync(amplitude)`.

---

## 6. Running the Frontend

To run the web interface locally:
```bash
cd web
npm install
npm run dev
```
The interface opens at `http://localhost:5173`. Click the ⚙️ icon to enter your `NINO_AUTH_TOKEN` (stored only in your browser's session).
