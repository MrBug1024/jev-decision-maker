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


def resolve_snapshot(source: str, cache_dir: str | None, revision: str | None) -> tuple[Path, str]:
    local = Path(source).expanduser()
    if local.is_dir():
        return local.resolve(), source

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


def import_model_class(config: dict[str, Any]):
    module = importlib.import_module(config["backbone_module"])
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
    cls = import_model_class(config)
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
        ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.onnx"),
        dirs_exist_ok=True,
    )


def save_model(model, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        destination,
        safe_serialization=True,
        max_shard_size="4GB",
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
    parser.add_argument("--source", default="akhilaaa3/Jev-Omni", help="Local snapshot or Hugging Face ID")
    parser.add_argument("--base-model", default="google/gemma-4-12B-it")
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

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f"Output already exists; choose a new path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    max_memory = parse_max_memory(args.max_memory)
    max_memory.setdefault("cpu", "64GiB")
    compute_dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16

    source, source_id = resolve_snapshot(args.source, args.cache_dir, args.revision)
    base, base_id = resolve_snapshot(args.base_model, args.cache_dir, args.base_revision)
    required = (source / "decision_config.json", source / "head.pt", source / "runtime_buffers.pt")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Jev-Omni source is missing: " + ", ".join(missing))

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
