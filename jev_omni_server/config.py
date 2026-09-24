"""Environment-backed settings for the standalone Jev-Omni service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _bool(name: str, default: bool) -> bool:
    value = _env(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = _env(name)
    try:
        return default if value is None else int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(name: str, default: float) -> float:
    value = _env(name)
    try:
        return default if value is None else float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _optional_float(name: str) -> float | None:
    value = _env(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


def _csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def _origins(value: str | None, fallback: str) -> list[str]:
    if not value:
        return [fallback]
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    index_html: Path
    database_path: Path
    model_id: str
    model_name: str
    model_kind: str
    model_inputs: tuple[str, ...]
    model_revision: str | None
    host: str
    port: int
    public_base_url: str
    cors_origins: list[str]
    cookie_secure: bool
    session_ttl: int
    password_iterations: int
    model_device: str
    device_map: str
    model_dtype: str
    quantization: str
    gpu_memory_reserve_gib: float
    cpu_memory_limit_gib: float | None
    model_cache_dir: Path | None
    offload_dir: Path
    max_concurrent_inference: int
    max_upload_mb: int
    skip_model_load: bool
    trust_remote_code: bool
    log_level: str

    @property
    def model_path(self) -> Path:
        return _path(self.model_id, self.base_dir / ".." / "models" / "jev-omni-int8").resolve()

    @property
    def mcp_resource_url(self) -> str:
        return f"{self.public_base_url}/mcp"


def load_settings() -> Settings:
    port = _int("JEV_OMNI_PORT", 8020)
    public_base_url = _env("JEV_OMNI_PUBLIC_URL", f"http://127.0.0.1:{port}").rstrip("/")
    inputs = _csv(_env("JEV_OMNI_MODEL_INPUTS"), ("text", "image", "audio", "video"))
    quantization = (_env("JEV_OMNI_QUANTIZATION", "auto") or "auto").lower()
    default_model = "../models/jev-omni-int4" if quantization in {"4", "4bit", "int4"} else "../models/jev-omni-int8"
    invalid = set(inputs) - {"text", "image", "audio", "video"}
    if invalid:
        raise ValueError("JEV_OMNI_MODEL_INPUTS contains unsupported values: " + ", ".join(sorted(invalid)))
    return Settings(
        base_dir=BASE_DIR,
        index_html=BASE_DIR / "index.html",
        database_path=_path(_env("JEV_OMNI_DB_PATH"), BASE_DIR.parent / "data" / "jev_gateway.db"),
        model_id=_env("JEV_OMNI_MODEL_PATH", default_model) or default_model,
        model_name=_env("JEV_OMNI_MODEL_NAME", "jev-omni-int8") or "jev-omni-int8",
        model_kind="jev_omni",
        model_inputs=inputs,
        model_revision=_env("JEV_OMNI_MODEL_REVISION"),
        host=_env("JEV_OMNI_HOST", "0.0.0.0") or "0.0.0.0",
        port=port,
        public_base_url=public_base_url,
        cors_origins=_origins(_env("JEV_OMNI_CORS_ORIGINS"), public_base_url),
        cookie_secure=_bool("JEV_OMNI_COOKIE_SECURE", False),
        session_ttl=_int("JEV_OMNI_SESSION_TTL", 60 * 60 * 24 * 30),
        password_iterations=_int("JEV_OMNI_PASSWORD_ITERATIONS", 310_000),
        model_device=(_env("JEV_OMNI_MODEL_DEVICE", "auto") or "auto").lower(),
        device_map=(_env("JEV_OMNI_DEVICE_MAP", "auto") or "auto").lower(),
        model_dtype=(_env("JEV_OMNI_MODEL_DTYPE", "auto") or "auto").lower(),
        quantization=quantization,
        gpu_memory_reserve_gib=_float("JEV_OMNI_GPU_MEMORY_RESERVE_GIB", 2.0),
        cpu_memory_limit_gib=_optional_float("JEV_OMNI_CPU_MEMORY_LIMIT_GIB") or 64.0,
        model_cache_dir=_path(_env("JEV_OMNI_MODEL_CACHE"), BASE_DIR / ".cache" / "huggingface")
        if _env("JEV_OMNI_MODEL_CACHE")
        else None,
        offload_dir=_path(_env("JEV_OMNI_OFFLOAD_DIR"), BASE_DIR / "model_offload"),
        max_concurrent_inference=max(1, _int("JEV_OMNI_MAX_CONCURRENT_INFERENCE", 1)),
        max_upload_mb=max(1, _int("JEV_OMNI_MAX_UPLOAD_MB", 50)),
        skip_model_load=_bool("JEV_OMNI_SKIP_MODEL_LOAD", False),
        trust_remote_code=_bool("JEV_OMNI_TRUST_REMOTE_CODE", True),
        log_level=(_env("JEV_OMNI_LOG_LEVEL", "info") or "info").lower(),
    )


settings = load_settings()
