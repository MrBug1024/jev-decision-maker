"""Local Jev-Omni bundle loader and multimodal classifier."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # Allows the console to import with model loading skipped.
    torch = None


def _dtype(name: str):
    import torch

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("model dtype must be bf16, fp16, or fp32")


def _device(value: Any):
    import torch

    if isinstance(value, torch.device):
        return value
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    if value == "disk":
        return torch.device("cpu")
    return torch.device(str(value))


def _mapped_device(model, *, last: bool):
    mapping = getattr(model, "hf_device_map", None) or {}
    if not mapping:
        return None
    values = list(mapping.values())
    if last:
        values = [value for name, value in mapping.items() if "lm_head" not in name]
    return _device(values[-1] if last and values else values[0])


def _input_device(model):
    import torch

    for getter in ("get_input_embeddings",):
        try:
            embedding = getattr(model, getter)()
            device = next(embedding.parameters()).device
            if device.type != "meta":
                return device
        except (AttributeError, StopIteration):
            pass
    return _mapped_device(model, last=False) or torch.device("cuda:0")


def _module_device(module, fallback):
    import torch

    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return fallback


def _quantization_config(bits: int | None, compute_dtype):
    if bits is None:
        return None
    from transformers import BitsAndBytesConfig

    if bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError("Only 4-bit and 8-bit Jev-Omni bundles are supported")


def _configure_cuda_for_reference_inference() -> None:
    """Match the official Space's CUDA settings for stable multimodal scores."""
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The official Space disables cuDNN SDPA because one CUDA image had a
    # version mismatch. The other SDPA backends remain available.
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def _max_memory(settings) -> dict[int | str, int | str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Jev-Omni requires a CUDA GPU")
    reserve = int(settings.gpu_memory_reserve_gib * 1024**3)
    limits: dict[int | str, int | str] = {}
    for index in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(index)
        limits[index] = max(free - reserve, 512 * 1024**2)
    if settings.cpu_memory_limit_gib is not None:
        limits["cpu"] = int(settings.cpu_memory_limit_gib * 1024**3)
    return limits


def _placement(settings, max_memory: dict[int | str, int | str]) -> dict[str, Any]:
    placement: dict[str, Any] = {"low_cpu_mem_usage": True}
    if settings.device_map in {"auto", "balanced", "balanced_low_0"}:
        placement["device_map"] = settings.device_map
        placement["max_memory"] = max_memory
        placement["offload_folder"] = str(settings.offload_dir)
    elif settings.device_map in {"single", "none"}:
        placement["device_map"] = {"": 0}
    else:
        raise ValueError("JEV_OMNI_DEVICE_MAP must be auto, balanced, balanced_low_0, single, or none")
    return placement


def _load_transformer(cls, path: Path, *, quantization_bits, compute_dtype, placement):
    kwargs: dict[str, Any] = {
        "dtype": compute_dtype,
        "attn_implementation": "sdpa",
        **placement,
    }
    quantization_config = _quantization_config(quantization_bits, compute_dtype)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return cls.from_pretrained(path, **kwargs).eval()


def _import_bundle_class(config: dict[str, Any], root: Path):
    module_name = config["backbone_module"]
    module_file = root / (module_name.replace(".", "/") + ".py")
    if module_file.exists():
        sys.path.insert(0, str(root))
    module = importlib.import_module(module_name)
    return getattr(module, config["backbone_class"])


def _load_processor(base_path: Path, *, trust_remote_code: bool):
    """Load the Gemma 4 processor while preserving the real import failure."""
    from transformers import AutoProcessor

    errors: list[Exception] = []
    for allow_remote_code in (False, trust_remote_code):
        if allow_remote_code is False and not errors:
            # Prefer the processor implementation shipped by Transformers when
            # the local config does not require dynamic remote code.
            pass
        try:
            return AutoProcessor.from_pretrained(
                base_path,
                trust_remote_code=allow_remote_code,
            )
        except Exception as exc:
            errors.append(exc)
            if allow_remote_code == trust_remote_code:
                break

    cause = errors[-1]
    details = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
    raise RuntimeError(
        "Could not load Gemma4UnifiedProcessor from the bundled base model. "
        "Install jev_omni_server/requirements.txt including torchvision, "
        "torchaudio, librosa, av, and torchcodec. "
        f"Attempts: {details}"
    ) from cause


def _find_backbone(model):
    for path in ("model.language_model", "language_model.model", "model.text_model", "model"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, "layers"):
            return path, node
    raise RuntimeError("Could not locate the Gemma text backbone in the base model")


class Head256:
    """The original FP32 Jev-Omni decision head."""

    def __new__(cls, hidden):
        import torch

        class _Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mu", torch.zeros(1, hidden))
                self.register_buffer("sd", torch.ones(1, hidden))
                self.linear = torch.nn.Linear(hidden, 256, dtype=torch.float32)

            def forward(self, features, counts):
                z = self.linear((features.float() - self.mu) / self.sd)
                indexes = torch.arange(256, device=z.device)[None]
                return z.masked_fill(indexes >= counts[:, None], -1e30)

        return _Head()


class LocalJevOmni:
    """A Jev-Omni classifier assembled entirely from a local bundle."""

    def __init__(self, model, head, processor, decoder, input_device, compute_dtype, manifest):
        import torch

        self.model = model
        self.head = head
        self.processor = processor
        self.decoder = decoder
        self.device = input_device
        self.compute_dtype = compute_dtype
        self.manifest = manifest
        self._capture: dict[str, Any] = {}
        decoder.register_forward_hook(
            lambda _module, _args, output: self._capture.__setitem__(
                "hidden",
                (output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0])[:, -1].float(),
            )
        )
        self._extra = (
            {"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(model.forward).parameters
            else {}
        )

    def _validate_multimodal_inputs(self, inputs: Any, modality: str) -> None:
        """Fail loudly when a processor call did not produce media features."""
        import torch

        if modality == "text":
            return

        required = {
            "image": "pixel_values",
            "video": "pixel_values_videos",
            "audio": "input_features",
        }[modality]
        payload = inputs.get(required)
        if not isinstance(payload, torch.Tensor) or payload.numel() == 0:
            raise RuntimeError(
                f"The processor did not create {required} for {modality} input. "
                "The uploaded media did not reach the Gemma multimodal path."
            )

        input_ids = inputs.get("input_ids")
        if modality == "image" and isinstance(input_ids, torch.Tensor):
            token_id = getattr(self.processor, "image_token_id", None)
            if token_id is None:
                token_id = getattr(getattr(self.model, "config", None), "image_token_id", None)
            if token_id is not None and not bool((input_ids == token_id).any().item()):
                raise RuntimeError(
                    "The processor created image pixels but no image placeholder token. "
                    "Check the bundled tokenizer/chat template files."
                )

    @classmethod
    def _from_unified(cls, root: Path, settings=None) -> "LocalJevOmni":
        """Load the official merged BF16 checkpoint shipped in ``unified/``."""
        import torch
        import transformers
        from transformers import AutoConfig

        root = root.resolve()
        unified_path = root / "unified"
        if settings is None:
            class Defaults:
                device_map = "auto"
                gpu_memory_reserve_gib = 2.0
                cpu_memory_limit_gib = None
                offload_dir = root / "offload"
                model_dtype = "bf16"
                trust_remote_code = True
                quantization = "bf16"

            settings = Defaults()
        requested = str(getattr(settings, "quantization", "bf16")).lower()
        if requested in {"4", "4bit", "int4", "8", "8bit", "int8"}:
            raise ValueError(
                "The official unified Jev-Omni checkpoint is full precision. "
                "Set JEV_OMNI_QUANTIZATION=bf16 when using a raw model directory."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Jev-Omni requires a CUDA GPU")

        dtype_name = getattr(settings, "model_dtype", "bf16")
        if dtype_name == "auto":
            dtype_name = "bf16"
        compute_dtype = _dtype(dtype_name)
        print(
            f"Jev-Omni runtime dtype: {dtype_name} (official unified checkpoint) ...",
            flush=True,
        )
        _configure_cuda_for_reference_inference()
        settings.offload_dir.mkdir(parents=True, exist_ok=True)
        placement = _placement(settings, _max_memory(settings))

        if not (unified_path / "config.json").exists():
            raise FileNotFoundError(f"Missing official unified model: {unified_path}")
        config = AutoConfig.from_pretrained(
            unified_path,
            trust_remote_code=settings.trust_remote_code,
        )
        architectures = getattr(config, "architectures", None) or []
        if not architectures:
            raise RuntimeError("The official unified model does not declare an architecture")
        model_cls = getattr(transformers, architectures[0], None)
        if model_cls is None:
            raise RuntimeError(f"Transformers has no architecture {architectures[0]!r}")

        print("Loading official unified Jev-Omni BF16 model ...", flush=True)
        model = _load_transformer(
            model_cls,
            unified_path,
            quantization_bits=None,
            compute_dtype=compute_dtype,
            placement=placement,
        )
        input_device = _input_device(model)
        decision_config = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
        head = Head256(decision_config["hidden_size"])
        head.load_state_dict(torch.load(root / "head.pt", map_location="cpu", weights_only=True))
        output_device = _mapped_device(model, last=True) or input_device
        head.to(output_device).eval()
        _, decoder = _find_backbone(model)
        processor = _load_processor(
            unified_path,
            trust_remote_code=settings.trust_remote_code,
        )
        model.eval()
        manifest = {
            "schema_version": 1,
            "model": "akhilaaa3/Jev-Omni",
            "source_model": "official-unified",
            "base_model": "google/gemma-4-12B-it",
            "quantization": {
                "bits": 0,
                "method": "none",
                "type": "bf16",
                "compute_dtype": dtype_name,
                "base_model": "none",
            },
        }
        return cls(model, head, processor, decoder, input_device, compute_dtype, manifest)

    @classmethod
    def from_bundle(cls, root: Path, settings=None) -> "LocalJevOmni":
        import torch
        import transformers
        from transformers import AutoConfig

        root = root.resolve()
        if not (root / "manifest.json").exists() and (root / "unified" / "config.json").exists():
            return cls._from_unified(root, settings)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        quant = manifest.get("quantization", {})
        bits = int(quant.get("bits", 0))
        if bits not in {4, 8}:
            raise ValueError("Bundle manifest must declare a 4-bit or 8-bit quantization")
        requested = str(getattr(settings, "quantization", "auto")).lower()
        if requested in {"4", "4bit", "int4"} and bits != 4:
            raise ValueError(f"Configured INT4 but the bundle contains INT{bits}")
        if requested in {"8", "8bit", "int8"} and bits != 8:
            raise ValueError(f"Configured INT8 but the bundle contains INT{bits}")
        if not torch.cuda.is_available():
            raise RuntimeError("Jev-Omni requires a CUDA GPU")
        if settings is None:
            class Defaults:
                device_map = "auto"
                gpu_memory_reserve_gib = 2.0
                cpu_memory_limit_gib = None
                offload_dir = root / "offload"
                model_dtype = quant.get("compute_dtype", "bf16")
                trust_remote_code = True

            settings = Defaults()
        dtype_name = settings.model_dtype
        if dtype_name == "auto":
            # The official Gemma 4 multimodal path runs in BF16. INT8
            # bitsandbytes layers may emit a BF16->FP16 cast warning, but
            # forcing the whole model to FP16 can produce non-finite values
            # in the 12B vision/text path. Keep FP16 only for the artifact
            # whose multimodal base was itself quantized.
            base_is_quantized = quant.get("base_model", "same") == "same"
            dtype_name = (
                "fp16"
                if bits == 8 and base_is_quantized
                else quant.get("compute_dtype", "bf16")
            )
        compute_dtype = _dtype(dtype_name)
        print(
            f"Jev-Omni runtime dtype: {dtype_name} "
            f"(base quantization: {quant.get('base_model', 'same')}) ...",
            flush=True,
        )
        _configure_cuda_for_reference_inference()
        settings.offload_dir.mkdir(parents=True, exist_ok=True)
        max_memory = _max_memory(settings)
        placement = _placement(settings, max_memory)

        base_path = root / "base_model"
        decoder_path = root / "backbone"
        if not (base_path / "config.json").exists():
            raise FileNotFoundError(f"Missing bundled base model: {base_path}")
        if not (decoder_path / "config.json").exists():
            raise FileNotFoundError(f"Missing bundled Jev-Omni backbone: {decoder_path}")

        base_config = AutoConfig.from_pretrained(base_path, trust_remote_code=settings.trust_remote_code)
        architectures = getattr(base_config, "architectures", None) or []
        if not architectures:
            raise RuntimeError("The bundled base model does not declare an architecture")
        base_cls = getattr(transformers, architectures[0], None)
        if base_cls is None:
            raise RuntimeError(f"Transformers has no architecture {architectures[0]!r}")

        base_bits = bits if quant.get("base_model", "same") == "same" else None
        print("Loading bundled Gemma multimodal base model ...", flush=True)
        base = _load_transformer(
            base_cls,
            base_path,
            quantization_bits=base_bits,
            compute_dtype=compute_dtype,
            placement=placement,
        )
        input_device = _input_device(base)

        decision_config = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
        decoder_cls = _import_bundle_class(decision_config, root)
        print("Loading bundled quantized Jev-Omni decoder ...", flush=True)
        decoder = _load_transformer(
            decoder_cls,
            decoder_path,
            quantization_bits=bits,
            compute_dtype=compute_dtype,
            placement=_placement(settings, _max_memory(settings)),
        )

        decoder_input_device = _input_device(decoder)
        buffers = torch.load(root / "runtime_buffers.pt", map_location="cpu", weights_only=True)
        for name, value in buffers.items():
            parent, _, attribute = name.rpartition(".")
            module = decoder.get_submodule(parent) if parent else decoder
            setattr(module, attribute, value.to(_module_device(module, decoder_input_device)))

        head = Head256(decision_config["hidden_size"])
        head.load_state_dict(torch.load(root / "head.pt", map_location="cpu", weights_only=True))
        decoder_output_device = _mapped_device(decoder, last=True) or decoder_input_device
        head.to(decoder_output_device).eval()

        path, old_backbone = _find_backbone(base)
        parent_path, _, name = path.rpartition(".")
        parent = base.get_submodule(parent_path) if parent_path else base
        setattr(parent, name, decoder)
        del old_backbone
        torch.cuda.empty_cache()

        processor = _load_processor(
            base_path,
            trust_remote_code=settings.trust_remote_code,
        )
        base.eval()
        return cls(base, head, processor, decoder, input_device, compute_dtype, manifest)

    @staticmethod
    def _video_frames(path: str | Path, count: int = 16):
        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(str(path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = sorted({int(round((total - 1) * (index + 0.5) / count)) for index in range(count)})
        frames, current = [], 0
        for index in wanted:
            while current < index and capture.grab():
                current += 1
            ok, frame = capture.read()
            current += 1
            if ok:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        capture.release()
        if not frames:
            raise ValueError(f"Could not decode video: {path}")
        return frames

    @staticmethod
    def _prompt(state: str, question: str, options: list[str]) -> str:
        choices = "\n".join(f"{index + 1}. {value}" for index, value in enumerate(options))
        return (
            f"{state}\n\n---\n\nQUESTION: {question}\n\nOPTIONS:\n{choices}\n\n"
            f"Reply with only the number of the correct option (1-{len(options)}).\n"
            "Output a single number and nothing else."
        )

    @torch.inference_mode() if torch is not None else (lambda function: function)
    def predict(
        self,
        *,
        state: str,
        question: str,
        options: list[str],
        media: str | Path | None = None,
        modality: str = "text",
        video_frames: int = 16,
    ) -> dict[str, Any]:
        import torch
        from PIL import Image

        if not 2 <= len(options) <= 256:
            raise ValueError("Jev-Omni needs 2-256 options")
        if modality not in {"text", "image", "audio", "video"}:
            raise ValueError("modality must be text, image, audio, or video")
        if modality != "text" and media is None:
            raise ValueError(f"{modality} inference requires media")

        content: list[dict[str, Any]] = []
        temporary: Path | None = None
        if modality == "image":
            with Image.open(media) as image:
                content.append({"type": "image", "image": image.convert("RGB")})
        elif modality == "video":
            content.extend({"type": "image", "image": frame} for frame in self._video_frames(media, video_frames))
        elif modality == "audio":
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            temporary = Path(handle.name)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(media), "-t", "30", "-ac", "1", "-ar", "16000", str(temporary)],
                check=True,
            )
            content.append({"type": "audio", "audio": str(temporary)})
        content.append({"type": "text", "text": self._prompt(state, question, options)})
        try:
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        self._validate_multimodal_inputs(inputs, modality)

        prepared = {
            key: value.to(self.device, dtype=self.compute_dtype) if torch.is_floating_point(value) else value.to(self.device)
            for key, value in inputs.items()
        }
        self._capture.clear()
        with torch.autocast("cuda", dtype=self.compute_dtype):
            self.model(**prepared, use_cache=False, **self._extra)
            hidden = self._capture.get("hidden")
            if hidden is None:
                raise RuntimeError("Jev-Omni decoder did not expose a final hidden state")
            if not torch.isfinite(hidden).all():
                raise RuntimeError(
                    "Jev-Omni produced non-finite hidden states. "
                    "Use JEV_OMNI_MODEL_DTYPE=bf16 for the multimodal base."
                )
            head_device = next(self.head.parameters()).device
            logits = self.head(
                hidden.to(head_device),
                torch.tensor([len(options)], device=head_device),
            )
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "Jev-Omni decision head produced non-finite logits. "
                    "Use JEV_OMNI_MODEL_DTYPE=bf16 and verify the bundle."
                )
            values_tensor = torch.softmax(logits[0, : len(options)].float(), dim=-1)
            if not torch.isfinite(values_tensor).all():
                raise RuntimeError("Jev-Omni produced non-finite probabilities")
            values = values_tensor.cpu().tolist()
        best = max(range(len(values)), key=values.__getitem__)
        return {
            "prediction": options[best],
            "prediction_index": best,
            "confidence": values[best],
            "probabilities": dict(zip(options, values)),
        }
