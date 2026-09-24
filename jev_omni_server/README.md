# Jev-Omni server

This is an independent MCP/console service for a locally saved Jev-Omni bundle. It does not modify or import the root `open-jev-deberta-v3-large` service.

It uses `../data/jev_gateway.db` by default, the same SQLite database used by `open_jev_server`. Therefore users, password hashes, sessions, and API Keys are shared across the two services. If the directories are deployed separately, set `OPEN_JEV_DB_PATH` and `JEV_OMNI_DB_PATH` to the same absolute path on a shared local filesystem.

For production, use the official merged BF16 checkpoint under `../models/raw/jev-omni`.
The `quantify/quantize_jev_omni.py` artifacts remain available for experiments, but
the current INT4/INT8 Jev decision backbone has a material option-position bias and
must not be treated as an accuracy-equivalent replacement.

Configure `.env` before starting the service:

```bash
cp .env.example .env
python server.py
```

The `.env` file controls the visible GPU and the model. The default example uses
`CUDA_VISIBLE_DEVICES=0`, `JEV_OMNI_MODEL_PATH=../models/raw/jev-omni`,
`JEV_OMNI_QUANTIZATION=bf16`, and `JEV_OMNI_MODEL_DTYPE=bf16`; keep these values
in `.env` instead of prefixing them on every startup command.

For an explicitly selected experiment, set `JEV_OMNI_QUANTIZATION=4bit` or
`8bit`; without an explicit model path this selects the matching quantized artifact.
`JEV_OMNI_MODEL_PATH` always takes precedence.

The service uses port `8020` by default and exposes the same console, SQLite account/key management, MCP endpoint, and `jev_decide` tool shape as the original service. It accepts text, image, audio, and video input in the browser test console. Audio conversion requires `ffmpeg` on `PATH`; video decoding uses OpenCV.

For reliable image/audio/video decisions, the bundle must have `quantization.base_model` set to `none`. Rebuild an older `base_model: same` bundle with `--base-quantization none`; the server logs a warning for the older low-memory artifact because quantizing the Gemma multimodal base can produce first-option position bias.

On startup, the service migrates the shared SQLite database by adding the user `role` column and the `api_call_logs` audit table. The existing account named `admin` is promoted to the administrator role automatically. Administrators can view all registered users and filter all Jev-Omni calls by user, Key, source, and status; regular users can only view their own calls. Audit records contain metadata only: no password, full API Key, prompt, or uploaded media is stored.

For a recommended `base_model: none` bundle, `JEV_OMNI_MODEL_DTYPE=auto` selects BF16 to match the official Gemma 4 multimodal path. bitsandbytes INT8 layers may log BF16-to-FP16 cast warnings; those warnings are expected. Set `JEV_OMNI_MODEL_DTYPE=bf16` explicitly if you want to pin the stable multimodal dtype. The old `base_model: same` low-memory artifact keeps the FP16 fallback, but is not recommended for multimodal accuracy.

The server never quantizes or downloads a model at startup. A production raw model
directory must contain `unified/config.json`, `unified/model.safetensors`,
`unified/processor_config.json`, `decision_config.json`, and `head.pt`. Quantized
artifacts instead contain `manifest.json`, `base_model/`, `backbone/`, `head.pt`,
and `runtime_buffers.pt`.

For a CPU-free production process, use one Uvicorn worker. The official BF16 model
requires a compatible Linux CUDA, PyTorch, and Transformers environment; the
quantized experimental bundles also require bitsandbytes.

SQLite sharing is intended for two services on the same machine. Do not place the SQLite file on NFS or use it as a cross-machine database; use PostgreSQL or another server database when the services run on different hosts.
