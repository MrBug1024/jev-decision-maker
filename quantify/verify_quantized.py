"""Validate a saved Jev-Omni quantization bundle and optionally run a probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_FILES = (
    "manifest.json",
    "decision_config.json",
    "head.pt",
    "runtime_buffers.pt",
    "backbone/config.json",
    "base_model/config.json",
)


def validate(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative in REQUIRED_FILES:
        if not (root / relative).exists():
            raise ValueError(f"Required artifact is missing: {relative}")
    quant = manifest.get("quantization", {})
    if quant.get("bits") not in {4, 8}:
        raise ValueError("manifest quantization.bits must be 4 or 8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--probe",
        type=Path,
        help="Optional JSONL file with state/question/options entries for a real inference probe",
    )
    args = parser.parse_args(argv)
    root = args.artifact.expanduser().resolve()
    manifest = validate(root)
    print(json.dumps({
        "artifact": str(root),
        "model": manifest.get("model"),
        "bits": manifest["quantization"]["bits"],
        "base_quantization": manifest["quantization"].get("base_model"),
        "file_count": len(manifest.get("files", [])),
    }, ensure_ascii=False, indent=2))

    if args.probe:
        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project_root))
        from jev_omni_server.omni_model import LocalJevOmni

        model = LocalJevOmni.from_bundle(root)
        for line_number, line in enumerate(args.probe.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            request = json.loads(line)
            result = model.predict(
                state=request["state"],
                question=request["question"],
                options=request["options"],
                media=request.get("media"),
                modality=request.get("modality", "text"),
            )
            print(json.dumps({"line": line_number, "result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
