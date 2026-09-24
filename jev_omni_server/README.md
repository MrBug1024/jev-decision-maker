# Jev-Omni server

This is an independent MCP/console service for a locally saved Jev-Omni bundle. It does not modify or import the root `open-jev-deberta-v3-large` service.

It uses `../data/jev_gateway.db` by default, the same SQLite database used by `open_jev_server`. Therefore users, password hashes, sessions, and API Keys are shared across the two services. If the directories are deployed separately, set `OPEN_JEV_DB_PATH` and `JEV_OMNI_DB_PATH` to the same absolute path on a shared local filesystem.

Create an artifact first with `quantify/quantize_jev_omni.py`, then configure `.env`:

```bash
cp .env.example .env
python server.py
```

The `.env` file controls the visible GPU and the bundled model. The default example uses `CUDA_VISIBLE_DEVICES=0`, `JEV_OMNI_MODEL_PATH=../models/jev-omni-int8`, and `JEV_OMNI_QUANTIZATION=8bit`; change those values there instead of prefixing them on every startup command.

Set `JEV_OMNI_QUANTIZATION=4bit` to make the default model path `../models/jev-omni-int4`; otherwise the default is `../models/jev-omni-int8`. `JEV_OMNI_MODEL_PATH` always takes precedence.

The service uses port `8020` by default and exposes the same console, SQLite account/key management, MCP endpoint, and `jev_decide` tool shape as the original service. It accepts text, image, audio, and video input in the browser test console. Audio conversion requires `ffmpeg` on `PATH`; video decoding uses OpenCV.

For reliable image/audio/video decisions, the bundle must have `quantization.base_model` set to `none`. Rebuild an older `base_model: same` bundle with `--base-quantization none`; the server logs a warning for the older low-memory artifact because quantizing the Gemma multimodal base can produce first-option position bias.

On startup, the service migrates the shared SQLite database by adding the user `role` column and the `api_call_logs` audit table. The existing account named `admin` is promoted to the administrator role automatically. Administrators can view all registered users and filter all Jev-Omni calls by user, Key, source, and status; regular users can only view their own calls. Audit records contain metadata only: no password, full API Key, prompt, or uploaded media is stored.

For a recommended `base_model: none` bundle, `JEV_OMNI_MODEL_DTYPE=auto` selects BF16 to match the official Gemma 4 multimodal path. bitsandbytes INT8 layers may log BF16-to-FP16 cast warnings; those warnings are expected. Set `JEV_OMNI_MODEL_DTYPE=bf16` explicitly if you want to pin the stable multimodal dtype. The old `base_model: same` low-memory artifact keeps the FP16 fallback, but is not recommended for multimodal accuracy.

The server never quantizes or downloads a model at startup. `JEV_OMNI_MODEL_PATH` must point to a complete bundle containing `manifest.json`, `base_model/`, `backbone/`, `head.pt`, and `runtime_buffers.pt`.

For a CPU-free production process, use one Uvicorn worker. The quantized bundle still requires a compatible Linux CUDA, PyTorch, Transformers, and bitsandbytes environment.

SQLite sharing is intended for two services on the same machine. Do not place the SQLite file on NFS or use it as a cross-machine database; use PostgreSQL or another server database when the services run on different hosts.
