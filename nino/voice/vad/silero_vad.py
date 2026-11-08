"""Silero VAD implementation using ONNX Runtime (zero PyTorch dependencies)."""

import logging
import math
from pathlib import Path
import numpy as np

from nino.voice.interfaces import VADDetector

logger = logging.getLogger("nino.voice.vad.silero")

SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


class SileroVAD(VADDetector):
    """Voice Activity Detector running the Silero VAD ONNX model via ONNX Runtime."""

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: str | Path = "data/models/vad/silero_vad.onnx",
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self._session = None
        self._state = None
        self._use_fallback = False

    def download_model_if_needed(self) -> Path:
        """Download silero_vad.onnx if not present locally."""
        if self.model_path.exists():
            return self.model_path

        import urllib.request
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Silero VAD ONNX model (~2MB) to %s", self.model_path)
        try:
            urllib.request.urlretrieve(SILERO_VAD_URL, self.model_path)
            logger.info("Silero VAD ONNX model downloaded successfully.")
        except Exception as e:
            logger.warning("Could not download Silero VAD ONNX model (%s). Using fallback VAD.", e)
            self._use_fallback = True

        return self.model_path

    def _ensure_session(self):
        if self._session is None and not self._use_fallback:
            try:
                import onnxruntime as ort
                path = self.download_model_if_needed()
                if path.exists():
                    opts = ort.SessionOptions()
                    opts.inter_op_num_threads = 1
                    opts.intra_op_num_threads = 1
                    opts.log_severity_level = 3  # Warning level
                    self._session = ort.InferenceSession(str(path), sess_options=opts)
                    self.reset()
                else:
                    self._use_fallback = True
            except Exception as e:
                logger.warning("Failed to initialize ONNX Runtime for Silero VAD: %s. Using energy VAD fallback.", e)
                self._use_fallback = True

    def reset(self) -> None:
        """Reset internal recurrent state."""
        if self._session is not None:
            # Check model input names to determine state shape (V4 vs V5)
            input_names = [inp.name for inp in self._session.get_inputs()]
            if "state" in input_names:
                self._state = np.zeros((2, 1, 128), dtype=np.float32)
            else:
                self._state = {
                    "h": np.zeros((2, 1, 64), dtype=np.float32),
                    "c": np.zeros((2, 1, 64), dtype=np.float32),
                }

    def speech_probability(self, frame_pcm: bytes) -> float:
        """Calculate speech probability for a 16kHz mono PCM frame."""
        if not frame_pcm:
            return 0.0

        # Convert 16-bit PCM bytes to float32
        audio_array = np.frombuffer(frame_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_array) == 0:
            return 0.0

        self._ensure_session()

        # Fallback to RMS energy if ONNX is unavailable
        if self._use_fallback or self._session is None:
            rms = np.sqrt(np.mean(audio_array ** 2))
            # Rough energy calibration: speech is typically > 0.015 RMS
            prob = float(min(1.0, max(0.0, (rms - 0.005) / 0.03)))
            return prob

        # Silero VAD expects chunk sizes of 512 samples for 16kHz
        # Pad or slice to 512
        chunk_len = 512
        if len(audio_array) < chunk_len:
            audio_array = np.pad(audio_array, (0, chunk_len - len(audio_array)))
        elif len(audio_array) > chunk_len:
            audio_array = audio_array[:chunk_len]

        inputs_dict = {"input": audio_array.reshape(1, chunk_len)}

        # Add sample rate if requested by model
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "sr" in input_names:
            inputs_dict["sr"] = np.array(self.sample_rate, dtype=np.int64)

        if isinstance(self._state, dict):
            inputs_dict["h"] = self._state["h"]
            inputs_dict["c"] = self._state["c"]
        elif self._state is not None:
            inputs_dict["state"] = self._state

        try:
            outs = self._session.run(None, inputs_dict)
            prob = float(outs[0][0][0] if len(outs[0].shape) > 1 else outs[0][0])
            if isinstance(self._state, dict):
                self._state["h"] = outs[1]
                self._state["c"] = outs[2]
            elif len(outs) > 1:
                self._state = outs[1]
            return prob
        except Exception as e:
            logger.debug("VAD inference error, using energy: %s", e)
            rms = np.sqrt(np.mean(audio_array ** 2))
            return float(min(1.0, max(0.0, (rms - 0.005) / 0.03)))

    def is_speech(self, frame_pcm: bytes) -> bool:
        """Check if frame speech probability exceeds threshold."""
        prob = self.speech_probability(frame_pcm)
        return prob >= self.threshold
