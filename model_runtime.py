"""JEV model loading with automatic CPU, single-GPU, and multi-GPU support."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer
from typed_decisions.encoder import Collator, DecisionEncoder
from typed_decisions.open_jev import OpenJev

from config import Settings


class ModelRuntime:
    """Own one JEV model instance and choose its placement at startup.

    ``device_map=auto`` is intentionally calculated from the GPUs visible to
    this process. No GPU index is hard-coded, so CUDA_VISIBLE_DEVICES can be
    used to control which cards are available on a deployment host.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: OpenJev | None = None
        self._inference_slots = threading.BoundedSemaphore(settings.max_concurrent_inference)
        self._status: dict[str, Any] = {
            "ready": False,
            "strategy": "not_loaded",
            "device": "not_loaded",
            "devices": [],
            "dtype": None,
            "error": None,
        }

    @staticmethod
    def _device(value: Any) -> torch.device:
        if isinstance(value, torch.device):
            return value
        if isinstance(value, int):
            return torch.device(f"cuda:{value}")
        if value == "disk":
            return torch.device("cpu")
        return torch.device(str(value))

    def _target_device(self) -> torch.device:
        mode = self.settings.model_device
        if mode not in {"auto", "cpu", "cuda"}:
            raise ValueError("OPEN_JEV_MODEL_DEVICE must be auto, cpu, or cuda")
        if mode == "cpu":
            return torch.device("cpu")
        if mode == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("OPEN_JEV_MODEL_DEVICE=cuda but CUDA is not available")
        if mode == "cuda" or (mode == "auto" and torch.cuda.is_available()):
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _dtype(self, target: torch.device) -> torch.dtype:
        value = self.settings.model_dtype
        explicit = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if value in explicit:
            return explicit[value]
        if value != "auto":
            raise ValueError(
                "OPEN_JEV_MODEL_DTYPE must be auto, float32, float16, or bfloat16"
            )
        if target.type == "cpu":
            return torch.float32
        capabilities = [
            torch.cuda.get_device_capability(index)
            for index in range(torch.cuda.device_count())
        ]
        # Some older CUDA/PyTorch combinations report BF16 support even when
        # the card cannot execute the required CUBLAS kernels reliably.
        if capabilities and all(major >= 8 for major, _ in capabilities):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        if capabilities and all(major >= 7 for major, _ in capabilities):
            return torch.float16
        return torch.float32

    def _max_memory(self) -> dict[int | str, int | str]:
        if not torch.cuda.is_available():
            return {"cpu": "32GiB"}
        reserve = int(self.settings.gpu_memory_reserve_gib * 1024**3)
        limits: dict[int | str, int | str] = {}
        for index in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(index)
            usable = max(free - reserve, 512 * 1024**2)
            limits[index] = usable
        if self.settings.cpu_memory_limit_gib is not None:
            limits["cpu"] = int(self.settings.cpu_memory_limit_gib * 1024**3)
        return limits

    def _resolve_model_dir(self) -> str:
        source = self.settings.model_id
        local_source = Path(source).expanduser()
        if not local_source.is_absolute():
            local_source = self.settings.base_dir / local_source
        if local_source.is_dir():
            return str(local_source)
        kwargs: dict[str, Any] = {}
        if self.settings.model_revision:
            kwargs["revision"] = self.settings.model_revision
        if self.settings.model_cache_dir:
            kwargs["cache_dir"] = str(self.settings.model_cache_dir)
        return snapshot_download(source, **kwargs)

    @staticmethod
    def _mapped_device(model: torch.nn.Module, *, last: bool) -> torch.device | None:
        mapping = getattr(model, "hf_device_map", None) or {}
        if not mapping:
            return None
        candidates: list[Any] = []
        for name, _ in model.named_modules():
            if name in mapping:
                if last and any(
                    marker in name.lower()
                    for marker in ("pooler", "classifier", "lm_head", "head")
                ):
                    continue
                candidates.append(mapping[name])
        if "" in mapping and not candidates:
            candidates.append(mapping[""])
        if not candidates:
            candidates = list(mapping.values())
        value = candidates[-1] if last else candidates[0]
        return ModelRuntime._device(value)

    @staticmethod
    def _input_device(model: torch.nn.Module) -> torch.device:
        try:
            embedding = model.get_input_embeddings()
            device = next(embedding.parameters()).device
            if device.type != "meta":
                return device
        except (AttributeError, StopIteration):
            pass
        return ModelRuntime._mapped_device(model, last=False) or torch.device("cpu")

    def _load_backbone(
        self,
        model_dir: str,
        config: dict[str, Any],
        target: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.nn.Module, torch.device, str]:
        kwargs: dict[str, Any] = {
            "attn_implementation": config.get("attn_implementation", "eager"),
            "dtype": dtype,
            "trust_remote_code": self.settings.trust_remote_code,
        }
        use_auto_map = (
            target.type == "cuda"
            and self.settings.device_map == "auto"
            and torch.cuda.device_count() > 1
        )
        if use_auto_map:
            self.settings.offload_dir.mkdir(parents=True, exist_ok=True)
            kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": self._max_memory(),
                    "low_cpu_mem_usage": True,
                    "offload_folder": str(self.settings.offload_dir),
                }
            )
            backbone = AutoModel.from_pretrained(model_dir, **kwargs)
            input_device = self._input_device(backbone)
            output_device = self._mapped_device(backbone, last=True) or input_device
            return backbone, output_device, "device_map_auto"

        if self.settings.device_map not in {"auto", "none", "single"}:
            raise ValueError("OPEN_JEV_DEVICE_MAP must be auto, none, or single")
        kwargs["low_cpu_mem_usage"] = True
        backbone = AutoModel.from_pretrained(model_dir, **kwargs)
        return backbone, target, "single_device"

    def load(self) -> OpenJev:
        target = self._target_device()
        dtype = self._dtype(target)
        model_dir = self._resolve_model_dir()
        with open(Path(model_dir) / "open_jev_config.json", encoding="utf-8") as handle:
            model_config = json.load(handle)

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            cache_dir=str(self.settings.model_cache_dir) if self.settings.model_cache_dir else None,
            trust_remote_code=self.settings.trust_remote_code,
        )
        backbone, output_device, strategy = self._load_backbone(
            model_dir, model_config, target, dtype
        )
        model = DecisionEncoder(backbone, model_config["hidden"], model_config["pool"])
        model.head.load_state_dict(load_file(str(Path(model_dir) / "head.safetensors")))

        if strategy == "device_map_auto":
            # The backbone returns its final hidden state on the last mapped
            # device; the decision head must live there as well.
            model.head.to(device=output_device, dtype=dtype)
            input_device = self._input_device(backbone)
        else:
            model.to(device=target, dtype=dtype)
            input_device = target
        model.eval()

        self.model = OpenJev(
            model,
            tokenizer,
            Collator(
                tokenizer,
                max_state_tokens=model_config.get("max_state_tokens", 256),
                max_len=model_config.get("max_len", 512),
            ),
            model_config,
            input_device,
        )
        mapped_devices = sorted(
            {
                str(value)
                for value in (getattr(backbone, "hf_device_map", {}) or {}).values()
            }
        )
        self._status = {
            "ready": True,
            "strategy": strategy,
            "device": ", ".join(mapped_devices) if mapped_devices else str(input_device),
            "devices": mapped_devices or [str(input_device)],
            "dtype": str(dtype).replace("torch.", ""),
            "error": None,
        }
        return self.model

    def decide(self, state: str, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError(self._status.get("error") or "JEV model is not ready")
        with self._inference_slots:
            return self.model.decide(state, questions)

    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def set_error(self, error: str) -> None:
        self._status = {**self._status, "ready": False, "error": error}
