# Nino Vision Subsystem Testing & Verification Guide

This document outlines the manual verification checklist, hardware requirements, local vision model setup, and live screen/camera testing procedures for Nino's Phase 6A vision subsystem.

---

## 1. Prerequisites & Vision Model Setup

### 1.1 Hardware Compatibility
- **CPU**: 13th Gen Intel Core i3-1305U or compatible x86_64 processor.
- **Memory**: Minimum 8 GB RAM (leaves $>6$ GB free when running Moondream).
- **GPU**: Integrated Intel UHD Graphics (runs on CPU/DirectML via Ollama).

### 1.2 Install Local Vision Model
Download the lightweight Moondream vision-language model in Ollama:
```powershell
ollama run moondream
```
To verify the model is available locally:
```powershell
ollama list
```
You should see `moondream:latest` (or `moondream:1.8b`, ~1.7 GB) listed.

---

## 2. Automated Test Suite Verification

The automated test suite runs 100% offline with synthetic in-memory images and mocked model transports:
- **`tests/test_vision_adapter.py`**: Tests `VisionAdapter` interface, Ollama generate payloads, text extraction, empty image rejection, and error handling.
- **`tests/test_vision_tools_confirmation.py`**: Verifies that `screenshot_capture` and `camera_snapshot` declare `PermissionTier.SENSITIVE`, trigger `confirmation_required`, and execute strictly after approval.
- **`tests/test_vision_temp_cleanup.py`**: Tests `TempImageManager` ensuring zero residual image files remain on disk after normal completion and exceptions.
- **`tests/test_vision_privacy.py`**: Validates that `PrivacyGateway` blocks outbound image data URIs, binary image bytes, and image parameters (`ALLOW_EXTERNAL_IMAGES=false`), and confirms `AuditLogger` records image capture metadata without image content.
- **`tests/test_vision_taint_injection.py`**: Simulates an image containing prompt injection text (`"ignore previous instructions"`), verifies text is wrapped in `<untrusted_reference ... taint="UNTRUSTED">`, and verifies taint escalation on reversible tools.
- **`tests/test_vision_api.py`**: Tests REST endpoints `/vision/upload`, `/vision/screenshot`, and `/vision/camera`.

Run all vision tests:
```powershell
.venv\Scripts\pytest tests/test_vision* -v
```

---

## 3. Manual Verification Checklist (Live Screen, Camera & Uploads)

### Test 1: Image Upload via REST API
Upload an image (e.g. a PNG or JPEG photo) to Nino's vision endpoint:
```powershell
curl -X POST http://127.0.0.1:8000/vision/upload `
  -H "Authorization: Bearer <AUTH_TOKEN>" `
  -F "file=@my_photo.jpg" `
  -F "prompt=What is in this picture?"
```
- [ ] Response status is HTTP 200 with JSON payload containing `status: "success"`, image dimensions, and a natural language `description`.
- [ ] Any visible text in the photo is returned in `extracted_text`, wrapped in `<untrusted_reference ... taint="UNTRUSTED">`.
- [ ] Inspect `data/tmp_vision/`: verify no image files remain on disk after the request completes.

### Test 2: Image Upload & Preview in Web Avatar UI
1. Launch the web avatar client:
   ```powershell
   cd web
   npm run dev
   ```
2. Open `http://127.0.0.1:5173` in your browser.
3. Click the 📷 icon next to the chat bar and select a local photo.
   - [ ] Attachment drawer appears showing image thumbnail and filename.
4. Type a question (e.g., *"What is this?"*) and press Enter or click ➤.
   - [ ] User message shows `📷 [filename]: What is this?`.
   - [ ] Avatar transitions to `thinking` state.
   - [ ] Assistant bubble responds with description and any extracted text.
   - [ ] Attachment drawer resets and input is cleared.

### Test 3: On-Demand Screenshot Capture (SENSITIVE Confirmation)
In the Nino CLI (`.venv\Scripts\python -m nino.cli`) or API:
```powershell
curl -X POST http://127.0.0.1:8000/vision/screenshot `
  -H "Authorization: Bearer <AUTH_TOKEN>" `
  -H "Content-Type: application/json" `
  -d '{"prompt": "Describe what is on my screen"}'
```
1. Verify initial response:
   - [ ] Status 200 returns `decision: "confirmation_required"`, `tool_name: "screenshot_capture"`, and a pending `confirmation.id`.
   - [ ] Screen capture was **not** performed before confirmation.
2. Approve the pending confirmation:
   ```powershell
   .venv\Scripts\python -m nino.cli confirmations approve <confirmation_id>
   ```
3. Re-execute request with `confirmation_id`:
   ```powershell
   curl -X POST http://127.0.0.1:8000/vision/screenshot `
     -H "Authorization: Bearer <AUTH_TOKEN>" `
     -H "Content-Type: application/json" `
     -d '{"prompt": "Describe what is on my screen", "confirmation_id": "<confirmation_id>"}'
   ```
   - [ ] Execution succeeds (`decision: "executed"`).
   - [ ] Output accurately describes windows or text visible on your desktop.
   - [ ] Confirm no screenshot files remain on disk in `data/tmp_vision/`.

### Test 4: On-Demand Camera Snapshot (SENSITIVE + Visual Indicator)
1. Request camera snapshot:
   ```powershell
   curl -X POST http://127.0.0.1:8000/vision/camera `
     -H "Authorization: Bearer <AUTH_TOKEN>" `
     -H "Content-Type: application/json" `
     -d '{"prompt": "What do you see?"}'
   ```
   - [ ] Returns `confirmation_required` with confirmation token.
   - [ ] Webcam hardware LED does **not** turn on yet.
2. Approve the confirmation and execute with `confirmation_id`:
   - [ ] Terminal logs display: `[📷 CAMERA ACTIVE] Activating camera device 0 for snapshot...`.
   - [ ] Web Avatar UI displays the red `🔴 CAMERA ACTIVE` pill in the HUD.
   - [ ] Physical webcam LED illuminates briefly for a single frame snapshot and turns off immediately.
   - [ ] Terminal logs display: `[📷 CAMERA DEACTIVATED] Captured single frame. Camera hardware released.`.
   - [ ] Camera hardware is fully released (no process holding camera lock).
   - [ ] VLM describes the captured frame.
   - [ ] No image files remain in `data/tmp_vision/`.

### Test 5: Audit Trail Verification
Inspect the append-only audit log in `logs/audit.jsonl`:
```powershell
Get-Content logs/audit.jsonl -Tail 10
```
- [ ] Each capture records `event_type: "image_capture"`, `capture_type` (`"screenshot"`, `"camera"`, or `"upload"`), and timestamp.
- [ ] Verify that `content` is recorded as `"[IMAGE_CONTENT_NEVER_LOGGED]"`.
- [ ] Confirm that zero base64 image data or binary pixels are written to the audit log.
