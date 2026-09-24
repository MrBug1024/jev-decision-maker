"""Runtime wrapper for a saved Jev-Omni quantization bundle."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any

from config import Settings
from omni_model import LocalJevOmni


class ModelRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: LocalJevOmni | None = None
        self._inference_slots = threading.BoundedSemaphore(settings.max_concurrent_inference)
        self._status: dict[str, Any] = {
            "ready": False,
            "strategy": "not_loaded",
            "device": "not_loaded",
            "devices": [],
            "dtype": None,
            "model_kind": "jev_omni",
            "configured_inputs": list(settings.model_inputs),
            "supported_inputs": [],
            "quantization": settings.quantization,
            "model_path": str(settings.model_path),
            "error": None,
        }

    @property
    def supported_inputs(self) -> tuple[str, ...]:
        return tuple(self.settings.model_inputs)

    def supports_input_type(self, input_type: str) -> bool:
        return input_type in self.supported_inputs

    def load(self) -> LocalJevOmni:
        if self.settings.model_device == "cpu":
            raise RuntimeError("Jev-Omni does not support CPU inference")
        model = LocalJevOmni.from_bundle(self.settings.model_path, self.settings)
        manifest = model.manifest
        base_quantization = manifest.get("quantization", {}).get("base_model")
        if base_quantization == "same":
            print(
                "WARNING: this bundle quantizes the Gemma multimodal base. "
                "Rebuild with --base-quantization none for reliable image/audio/video inference.",
                flush=True,
            )
        devices = sorted(
            {
                str(value)
                for module in (model.model, model.decoder)
                for value in (getattr(module, "hf_device_map", {}) or {}).values()
            }
        )
        if not devices:
            devices = [str(model.device)]
        self.model = model
        self._status = {
            "ready": True,
            "strategy": f"{self.settings.device_map}_quantized",
            "device": str(model.device),
            "devices": devices,
            "dtype": str(model.compute_dtype).replace("torch.", ""),
            "head_device": str(next(model.head.parameters()).device),
            "model_kind": "jev_omni",
            "configured_inputs": list(self.settings.model_inputs),
            "supported_inputs": list(self.supported_inputs),
            "quantization": manifest.get("quantization", {}),
            "model_path": str(self.settings.model_path),
            "error": None,
        }
        return model

    @staticmethod
    def _score(values: list[float]) -> float:
        denominator = max(1, len(values) - 1)
        return sum(index * value for index, value in enumerate(values)) / denominator

    def _one(self, state: str, question: dict[str, Any], media: str | None, modality: str) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(self._status.get("error") or "Jev-Omni is not ready")
        kind = question["type"]
        options = list(question.get("options") or [])
        if kind == "noul":
            options = ["Yes", "No"]
        prediction = self.model.predict(
            state=state,
            question=question["instructions"],
            options=options,
            media=media,
            modality=modality,
        )
        if kind == "choice":
            return {
                "choice": prediction["prediction"],
                "probabilities": prediction["probabilities"],
                "confidence": prediction["confidence"],
            }
        if kind == "score":
            values = [prediction["probabilities"][option] for option in options]
            return {
                "score": self._score(values),
                "probabilities": prediction["probabilities"],
                "confidence": prediction["confidence"],
            }
        return {
            "noul": prediction["probabilities"]["Yes"],
            "probabilities": prediction["probabilities"],
            "confidence": prediction["confidence"],
        }

    def decide(
        self,
        state: str,
        questions: list[dict[str, Any]],
        *,
        modality: str = "text",
        media: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.supports_input_type(modality):
            raise RuntimeError(f"Jev-Omni does not expose {modality} input")
        with self._inference_slots:
            return [self._one(state, question, media, modality) for question in questions]

    def test_decide(
        self,
        input_type: str,
        situation: str,
        questions: list[dict[str, Any]],
        media: bytes | None = None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        del content_type
        temporary_path: Path | None = None
        if input_type != "text":
            if media is None:
                raise ValueError(f"{input_type} input requires a media file")
            suffix = Path(filename or "upload.bin").suffix or ".bin"
            handle = tempfile.NamedTemporaryFile(prefix="jev-omni-", suffix=suffix, delete=False)
            try:
                handle.write(media)
                handle.flush()
            finally:
                handle.close()
            temporary_path = Path(handle.name)
        try:
            return self.decide(
                situation,
                questions,
                modality=input_type,
                media=str(temporary_path) if temporary_path else None,
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def set_error(self, error: str) -> None:
        self._status = {
            **self._status,
            "ready": False,
            "supported_inputs": list(self.supported_inputs),
            "error": error,
        }
