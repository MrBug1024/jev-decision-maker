"""Build a self-contained INT8 or INT4 Jev-Omni model bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPO = "akhilaaa3/Jev-Omni"
DEFAULT_BASE_REPO = "google/gemma-4-12B-it"


def resolve_project_path(value: str | None, *, for_output: bool = False) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    current = (Path.cwd() / path).resolve()
    project = (PROJECT_ROOT / path).resolve()
    if current.exists():
        return str(current)
    project_relative = path.parts and path.parts[0].lower() in {
        "models",
        "data",
        "quantify",
        "jev_omni_server",
        "open_jev_server",
    }
    if project.exists() or for_output or project_relative:
        return str(project)
    return str(current)


def parse_max_memory(values: list[str]) -> dict[int | str, str]:
    result: dict[int | str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--max-memory must use INDEX=VALUE, for example 0=28GiB")
        key, limit = value.split("=", 1)
        if not limit.strip():
            raise ValueError(f"Empty memory limit in {value!r}")
        result["cpu" if key.lower() == "cpu" else int(key)] = limit.strip()
    return result


def _looks_like_local_path(source: str) -> bool:
    raw = Path(source).expanduser()
    return (
        raw.is_absolute()
        or source.startswith((".", "~", "/", "\\"))
        or len(raw.parts) > 2
        or source.startswith(("models/", "models\\"))
    )


def _has_weight_file(root: Path) -> bool:
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
    return any(path.is_file() and path.suffix in weight_suffixes for path in root.rglob("*"))


def resolve_snapshot(
    source: str,
    cache_dir: str | None,
    revision: str | None,
    *,
    download_repo: str | None = None,
    label: str,
    required_files: tuple[str, ...] = (),
    require_weights: bool = False,
) -> tuple[Path, str]:
    local_value = resolve_project_path(source)
    local = Path(local_value or source)
    if local.is_dir():
        complete = all((local / item).exists() for item in required_files)
        if require_weights:
            complete = complete and _has_weight_file(local)
        if complete:
            return local.resolve(), source
        if download_repo is None:
            return local.resolve(), source
        print(
            f"Resuming incomplete {label} model download in {local} ...",
            flush=True,
        )

    if _looks_like_local_path(source):
        if download_repo is None:
            raise FileNotFoundError(
                f"Local {label} directory does not exist: {source!r}. "
                f"Checked {local} and the current working directory {Path.cwd()}."
            )

        if local.exists() and not local.is_dir():
            raise FileNotFoundError(
                f"Local {label} path is not a directory: {local}"
            )

        local.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading {label} model {download_repo} to {local} ...",
            flush=True,
        )
        from huggingface_hub import snapshot_download

        kwargs: dict[str, Any] = {
            "repo_id": download_repo,
            "local_dir": str(local),
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if revision:
            kwargs["revision"] = revision
        try:
            snapshot_download(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {label} model {download_repo!r} to {local}. "
                "Check network access, Hugging Face authentication, and disk space."
            ) from exc
        return local.resolve(), download_repo

    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(source, **kwargs)).resolve(), source


def build_quantization_config(bits: int, compute_dtype):
    from transformers import BitsAndBytesConfig

    if bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def import_model_class(config: dict[str, Any], module_root: Path | None = None):
    module_name = config["backbone_module"]
    module_file = None
    if module_root is not None:
        module_file = module_root / (module_name.replace(".", "/") + ".py")
    if module_file is not None and module_file.exists():
        sys.path.insert(0, str(module_root))
    module = importlib.import_module(module_name)
    return getattr(module, config["backbone_class"])


def load_model(
    source: Path,
    *,
    bits: int | None,
    compute_dtype,
    device_map: str,
    max_memory: dict[int | str, str],
):
    import torch

    config = json.loads((source / "decision_config.json").read_text(encoding="utf-8"))
    cls = import_model_class(config, source)
    kwargs: dict[str, Any] = {
        "dtype": compute_dtype,
        "attn_implementation": "sdpa",
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if max_memory:
        kwargs["max_memory"] = max_memory
    if bits is not None:
        kwargs["quantization_config"] = build_quantization_config(bits, compute_dtype)
    return cls.from_pretrained(source / "backbone", **kwargs).eval()


def load_base_model(
    source: Path,
    *,
    bits: int | None,
    compute_dtype,
    device_map: str,
    max_memory: dict[int | str, str],
):
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(source)
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        raise RuntimeError("Gemma base model does not declare an architecture")
    cls = getattr(transformers, architectures[0], None)
    if cls is None:
        raise RuntimeError(f"Transformers has no architecture {architectures[0]!r}")
    kwargs: dict[str, Any] = {
        "dtype": compute_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if max_memory:
        kwargs["max_memory"] = max_memory
    if bits is not None:
        kwargs["quantization_config"] = build_quantization_config(bits, compute_dtype)
    return cls.from_pretrained(source, **kwargs).eval()


def copy_bundle_without_backbone(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("backbone"),
        dirs_exist_ok=True,
    )


def copy_base_bundle(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.safetensors.index.json",
            "*.bin.index.json",
            "*.onnx",
        ),
        dirs_exist_ok=True,
    )


def _state_dict_for_saving(model):
    import torch

    state_dict = model.state_dict()
    if any(
        isinstance(value, torch.Tensor) and value.device.type == "meta"
        for value in state_dict.values()
    ):
        try:
            from accelerate.utils.modeling import get_state_dict_from_offloaded_model
        except ImportError as exc:
            raise RuntimeError(
                "The quantized model contains meta tensors after CPU offload, "
                "but this Accelerate version cannot materialize its state_dict. "
                "Upgrade accelerate and rerun the quantization."
            ) from exc
        state_dict = get_state_dict_from_offloaded_model(model)

    materialized = {}
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Model state entry {name!r} is not a tensor")
        if value.device.type == "meta":
            raise RuntimeError(f"Model state entry {name!r} is still on the meta device")
        materialized[name] = value.detach().cpu()
    return materialized


def _save_torch_state_dict(state_dict, destination: Path, *, max_shard_bytes: int) -> None:
    import torch

    shards: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    current_bytes = 0
    for name, value in state_dict.items():
        value_bytes = value.numel() * value.element_size()
        if current and current_bytes + value_bytes > max_shard_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = value
        current_bytes += value_bytes
    if current:
        shards.append(current)

    if len(shards) == 1:
        torch.save(shards[0], destination / "pytorch_model.bin")
        return

    total = len(shards)
    weight_map: dict[str, str] = {}
    total_bytes = 0
    for index, shard in enumerate(shards, 1):
        filename = f"pytorch_model-{index:05d}-of-{total:05d}.bin"
        torch.save(shard, destination / filename)
        for name, value in shard.items():
            weight_map[name] = filename
            total_bytes += value.numel() * value.element_size()
    (destination / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_model(model, destination: Path) -> None:
    """Save without Transformers' tied-weight pass, which breaks bnb INT8 SCB."""
    destination.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "config") or not hasattr(model.config, "save_pretrained"):
        raise TypeError("The loaded model does not expose a Hugging Face config")

    print("Materializing quantized state_dict on CPU for checkpoint export ...", flush=True)
    state_dict = _state_dict_for_saving(model)
    model.config.save_pretrained(destination)
    _save_torch_state_dict(
        state_dict,
        destination,
        max_shard_bytes=4 * 1024**3,
    )


def file_inventory(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                }
            )
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path, args: argparse.Namespace, source_id: str, base_id: str) -> None:
    manifest = {
        "schema_version": 1,
        "model": "akhilaaa3/Jev-Omni",
        "source_model": source_id,
        "base_model": base_id,
        "quantization": {
            "bits": args.bits,
            "method": "bitsandbytes",
            "type": "nf4" if args.bits == 4 else "llm.int8",
            "compute_dtype": args.compute_dtype,
            "double_quant": args.bits == 4,
            "base_model": args.base_quantization,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": __import__("torch").__version__,
        },
        "files": file_inventory(root),
    }
    bitsandbytes = importlib.util.find_spec("bitsandbytes")
    if bitsandbytes:
        import bitsandbytes

        manifest["runtime"]["bitsandbytes"] = getattr(bitsandbytes, "__version__", "unknown")
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name in ("decision_config.json", "config.json"):
        path = root / name
        if path.exists():
            (root / f"{name}.sha256").write_text(
                f"{sha256_file(path)}  {name}\n",
                encoding="ascii",
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE_REPO,
        help="Local snapshot or Hugging Face ID; a missing models/raw/jev-omni path is downloaded automatically",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_REPO,
        help="Local Gemma snapshot or Hugging Face ID; a missing models/raw/gemma-4-12B-it path is downloaded automatically",
    )
    parser.add_argument("--output", required=True, help="New output directory; it must not already exist")
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--base-quantization", choices=("same", "none"), default="same")
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device-map", default="auto", choices=("auto", "balanced", "balanced_low_0"))
    parser.add_argument("--max-memory", action="append", default=[], metavar="INDEX=VALUE")
    parser.add_argument("--cache-dir")
    parser.add_argument("--revision")
    parser.add_argument("--base-revision")
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        parser.error("Jev-Omni quantization requires a CUDA-enabled PyTorch runtime")
    if args.compute_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("BF16 is not supported by the visible CUDA device; use --compute-dtype fp16")

    output = Path(resolve_project_path(args.output, for_output=True) or args.output).resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    max_memory = parse_max_memory(args.max_memory)
    max_memory.setdefault("cpu", "64GiB")
    compute_dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16

    cache_dir = resolve_project_path(args.cache_dir)
    try:
        source, source_id = resolve_snapshot(
            args.source,
            cache_dir,
            args.revision,
            download_repo=DEFAULT_SOURCE_REPO,
            label="Jev-Omni source",
            required_files=(
                "decision_config.json",
                "head.pt",
                "runtime_buffers.pt",
                "backbone/config.json",
            ),
        )
        base, base_id = resolve_snapshot(
            args.base_model,
            cache_dir,
            args.base_revision,
            download_repo=DEFAULT_BASE_REPO,
            label="Gemma base",
            required_files=("config.json",),
            require_weights=True,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    required = (
        source / "decision_config.json",
        source / "head.pt",
        source / "runtime_buffers.pt",
        source / "backbone" / "config.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Jev-Omni source is missing: " + ", ".join(missing))
    if not _has_weight_file(base):
        parser.error(
            "Gemma base model is missing weight files under "
            f"{base}; rerun the command to resume its download"
        )

    temp = Path(tempfile.mkdtemp(prefix=f"{output.name}.", dir=output.parent))
    try:
        copy_bundle_without_backbone(source, temp)
        copy_base_bundle(base, temp / "base_model")

        print(f"Loading Jev-Omni backbone as INT{args.bits} ...", flush=True)
        decoder = load_model(
            source,
            bits=args.bits,
            compute_dtype=compute_dtype,
            device_map=args.device_map,
            max_memory=max_memory,
        )
        save_model(decoder, temp / "backbone")
        del decoder
        torch.cuda.empty_cache()

        print(f"Loading Gemma base model ({args.base_quantization}) ...", flush=True)
        base_model = load_base_model(
            base,
            bits=args.bits if args.base_quantization == "same" else None,
            compute_dtype=compute_dtype,
            device_map=args.device_map,
            max_memory=max_memory,
        )
        save_model(base_model, temp / "base_model")
        del base_model
        torch.cuda.empty_cache()

        write_manifest(temp, args, source_id, base_id)
        os.replace(temp, output)
        print(f"Created self-contained bundle: {output}")
        return 0
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
