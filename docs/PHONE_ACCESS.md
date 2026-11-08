# Phone Access & Mobile Setup Guide: Nino Personal AI Assistant

## 1. Overview & Zero-Exposure Model

Nino is designed to be fully accessible from your mobile phone (for voice conversations, 3D avatar interactions, briefings, and reminders) **without ever opening ports on your router or exposing your assistant to the public internet**.

### Core Invariants
- **Local-Only Binding**: Nino binds exclusively to `127.0.0.1` and your encrypted Tailscale mesh network.
- **Mutual WireGuard Encryption**: Tailscale establishes point-to-point WireGuard tunnels between your phone and your host computer.
- **Secure Origin HTTPS for Microphones**: Mobile operating systems (iOS Safari and Android Chrome) strictly require an HTTPS origin (or `localhost`) to grant microphone access (`navigator.mediaDevices.getUserMedia`). Tailscale provides free, automatic TLS certificates via **Tailscale Serve**.
- **Brute-Force Lockout Defense**: Nino enforces IP-based rate limiting and lockout (`AuthLockoutManager`). 5 failed authentication attempts result in an automatic 5-minute lockout.

---

## 2. Step-by-Step Setup with Tailscale

### Step 1: Install Tailscale on Both Devices
1. Install Tailscale on your host machine: [https://tailscale.com/download](https://tailscale.com/download).
2. Install the Tailscale app on your phone (available on Apple App Store and Google Play).
3. Sign in to the same Tailscale account on both devices. They will now share a private encrypted network (tailnet).
4. Note your computer's Tailscale machine name (e.g. `my-pc.tailnet-xyz.ts.net`) or Tailscale IP (e.g. `100.x.y.z`).

### Step 2: Enable Tailscale Serve (HTTPS)
To enable the microphone in your mobile browser, route traffic through Tailscale's built-in HTTPS proxy:

```powershell
# In PowerShell (run on host machine):
# 1. Forward HTTPS port 443 to the Nino Web Avatar frontend (port 5173)
tailscale serve --bg 5173

# 2. Forward WebSocket and REST backend traffic (port 8000)
tailscale serve --bg /api http://127.0.0.1:8000
tailscale serve --bg /ws http://127.0.0.1:8000/ws
```

### Step 3: Access from Your Phone
Open your mobile browser and navigate to:
```text
https://my-pc.tailnet-xyz.ts.net/?token=your_auth_token_here
```
- The token is automatically saved into your mobile browser's session storage.
- The 3D animated avatar loads instantly.
- Tap the 🎙️ **PTT** button to speak with Nino; your browser will prompt for microphone permission once.

---

## 3. Installing Nino as a Progressive Web App (PWA)

Nino includes a full PWA manifest and service worker:
- **On iOS (Safari)**:
  1. Open `https://my-pc.tailnet-xyz.ts.net/` in Safari.
  2. Tap the **Share** button (box with upward arrow).
  3. Scroll down and tap **Add to Home Screen**.
  4. Launch Nino from your home screen as a full-screen native app.
- **On Android (Chrome)**:
  1. Open the URL in Chrome.
  2. Tap the three-dot menu icon in the upper right.
  3. Tap **Install app** (or **Add to Home screen**).

---

## 4. Private Push Notifications over Tailscale (`ntfy`)

To receive proactive reminders and daily morning briefings directly on your lock screen without relying on Google or Apple cloud tracking:

### 1. Run a Local `ntfy` Server
You can run a lightweight, open-source `ntfy` server in Docker on your host machine:
```bash
docker run -d --name ntfy -p 8080:80 binwiederhier/ntfy serve
```

### 2. Configure Nino (`.env`)
```env
NTFY_URL=http://100.x.y.z:8080
NTFY_TOPIC=nino_alerts_mysecretcode
NTFY_TOKEN=
```

### 3. Install the `ntfy` App on Your Phone
1. Download the `ntfy` app (Android via F-Droid / Google Play, or iOS via App Store).
2. Tap **Add subscription**.
3. Set the server URL to your Tailscale machine (`http://100.x.y.z:8080`) and topic name (`nino_alerts_mysecretcode`).
4. Whenever a reminder or briefing is announced, it appears on your phone's notification center in real time.

---

## 5. Security & Authentication Lockout

- **Constant-Time Verification**: All endpoints enforce `secrets.compare_digest` Bearer token authentication.
- **Failed Attempt Lockout**:
  - `AUTH_MAX_FAILED_ATTEMPTS=5` (default)
  - `AUTH_LOCKOUT_SECONDS=300` (5 minutes)
  - After 5 unsuccessful attempts from any IP address (including phone or local network), all subsequent requests from that IP are blocked with `HTTP 429 Too Many Requests`.
