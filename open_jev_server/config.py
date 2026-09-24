"""Environment-backed configuration for the JEV Model Gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)


def _env(name: str, default: str | None = None, *aliases: str) -> str | None:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
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


def _origins(value: str | None, fallback: str) -> list[str]:
    if not value:
        return [fallback]
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


def _csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


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
    def mcp_resource_url(self) -> str:
        return f"{self.public_base_url}/mcp"


def load_settings() -> Settings:
    port = _int("OPEN_JEV_PORT", 8019)
    public_base_url = _env("OPEN_JEV_PUBLIC_URL", f"http://127.0.0.1:{port}").rstrip("/")
    model_cache = _env("OPEN_JEV_MODEL_CACHE")
    model_inputs = _csv(_env("OPEN_JEV_MODEL_INPUTS"), ("text",))
    invalid_inputs = set(model_inputs) - {"text", "image", "audio", "video"}
    if invalid_inputs:
        raise ValueError(
            "OPEN_JEV_MODEL_INPUTS contains unsupported values: "
            + ", ".join(sorted(invalid_inputs))
        )
    model_kind = _env("OPEN_JEV_MODEL_KIND", "typed_decisions").lower()
    if not model_kind:
        raise ValueError("OPEN_JEV_MODEL_KIND must not be empty")
    return Settings(
        base_dir=BASE_DIR,
        index_html=BASE_DIR / "index.html",
        database_path=_path(_env("OPEN_JEV_DB_PATH"), BASE_DIR.parent / "data" / "jev_gateway.db"),
        model_id=_env(
            "OPEN_JEV_MODEL_ID",
            "com-kotobalabs/open-jev-deberta-v3-large",
            "MODEL_ID",
        ),
        model_name=_env(
            "OPEN_JEV_MODEL_NAME",
            "open-jev-deberta-v3-large",
            "MODEL_NAME",
        ),
        model_kind=model_kind,
        model_inputs=model_inputs,
        model_revision=_env("OPEN_JEV_MODEL_REVISION"),
        host=_env("OPEN_JEV_HOST", "0.0.0.0"),
        port=port,
        public_base_url=public_base_url,
        cors_origins=_origins(_env("OPEN_JEV_CORS_ORIGINS"), public_base_url),
        cookie_secure=_bool("OPEN_JEV_COOKIE_SECURE", False),
        session_ttl=_int("OPEN_JEV_SESSION_TTL", 60 * 60 * 24 * 30),
        password_iterations=_int("OPEN_JEV_PASSWORD_ITERATIONS", 310_000),
        model_device=_env("OPEN_JEV_MODEL_DEVICE", "auto").lower(),
        device_map=_env("OPEN_JEV_DEVICE_MAP", "auto").lower(),
        model_dtype=_env("OPEN_JEV_MODEL_DTYPE", "auto").lower(),
        gpu_memory_reserve_gib=_float("OPEN_JEV_GPU_MEMORY_RESERVE_GIB", 2.0),
        cpu_memory_limit_gib=_optional_float("OPEN_JEV_CPU_MEMORY_LIMIT_GIB"),
        model_cache_dir=_path(model_cache, BASE_DIR / ".cache" / "huggingface") if model_cache else None,
        offload_dir=_path(_env("OPEN_JEV_OFFLOAD_DIR"), BASE_DIR / "model_offload"),
        max_concurrent_inference=max(1, _int("OPEN_JEV_MAX_CONCURRENT_INFERENCE", 1)),
        max_upload_mb=max(1, _int("OPEN_JEV_MAX_UPLOAD_MB", 50)),
        skip_model_load=_bool("OPEN_JEV_SKIP_MODEL_LOAD", False),
        trust_remote_code=_bool("OPEN_JEV_TRUST_REMOTE_CODE", False),
        log_level=_env("OPEN_JEV_LOG_LEVEL", "info").lower(),
    )


settings = load_settings()
