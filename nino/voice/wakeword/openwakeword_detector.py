"""OpenWakeWord detector implementation for on-device wake phrase detection."""

import logging
from typing import Optional
import numpy as np

from nino.voice.interfaces import WakeWordDetector

logger = logging.getLogger("nino.voice.wakeword")


class OpenWakeWordDetector(WakeWordDetector):
    """Wake word detector powered by openWakeWord using ONNX Runtime."""

    def __init__(
        self,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        inference_framework: str = "onnx",
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.inference_framework = inference_framework
        self._model = None
        self._active_key = None

    def _ensure_model(self):
        if self._model is None:
            import openwakeword
            logger.info("Initializing openWakeWord model '%s' (%s)", self.model_name, self.inference_framework)
            self._model = openwakeword.Model(
                wakeword_models=[self.model_name],
                inference_framework=self.inference_framework,
            )
            # Find the internal dictionary key matching the model
            for k in self._model.models.keys():
                if self.model_name.lower() in k.lower():
                    self._active_key = k
                    break
            if not self._active_key and self._model.models:
                self._active_key = list(self._model.models.keys())[0]

        return self._model

    def reset(self) -> None:
        """Reset internal prediction buffers."""
        if self._model is not None:
            self._model.reset()

    def process_frame(self, frame_pcm: bytes) -> tuple[bool, float, Optional[str]]:
        """Process 16-bit linear PCM audio frame (typically 1280 samples / 80ms at 16kHz).

        Returns:
            (detected: bool, score: float, wake_word_name: str | None)
        """
        if not frame_pcm:
            return False, 0.0, None

        audio_int16 = np.frombuffer(frame_pcm, dtype=np.int16)
        if len(audio_int16) == 0:
            return False, 0.0, None

        try:
            model = self._ensure_model()
            prediction = model.predict(audio_int16)
            score = 0.0
            if self._active_key and self._active_key in prediction:
                score = float(prediction[self._active_key])
            elif prediction:
                # Max score across all loaded models
                score = float(max(prediction.values()))

            detected = score >= self.threshold
            if detected:
                logger.info("Wake word '%s' detected with score %.3f (threshold=%.3f)", self.model_name, score, self.threshold)
                return True, score, self.model_name

            return False, score, self.model_name
        except Exception as e:
            logger.debug("Wake word detection error: %s", e)
            return False, 0.0, None
