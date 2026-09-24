"""Run a small image-order bias probe against a saved Jev-Omni bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--state", default="请只根据上传的图片作答，不要根据其他信息猜测。")
    parser.add_argument("--question", default="图片中的圆形是什么颜色？")
    parser.add_argument("--first", default="蓝色")
    parser.add_argument("--second", default="红色")
    parser.add_argument("--modality", choices=("image",), default="image")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from jev_omni_server.omni_model import LocalJevOmni

    artifact = args.artifact.expanduser().resolve()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        parser.error(f"Image does not exist: {image}")

    model = LocalJevOmni.from_bundle(artifact)
    cases = (
        [args.first, args.second],
        [args.second, args.first],
    )
    results = []
    for options in cases:
        result = model.predict(
            state=args.state,
            question=args.question,
            options=options,
            media=image,
            modality=args.modality,
        )
        results.append({"options": options, "result": result})

    print(json.dumps({"artifact": str(artifact), "image": str(image), "cases": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
