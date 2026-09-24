# Jev-Omni quantization

This directory creates complete, reusable Jev-Omni bundles. The source model and Gemma base model are copied into the output, while the decoder backbone and optionally the base model are saved with bitsandbytes INT8 or NF4 INT4 weights.

Run this on the Linux CUDA host that will perform the conversion. Relative model and output paths are resolved from the project root, so the commands work both from the repository root and from inside `quantify/`. The output directory must be new; the command never overwrites an existing artifact.

If `models/raw/jev-omni` or `models/raw/gemma-4-12B-it` does not exist, the command downloads the default repositories `akhilaaa3/Jev-Omni` and `google/gemma-4-12B-it` into those directories first. An interrupted download can be rerun; the Hugging Face download is resumed and the existing files are reused.

```bash
python -m pip install -r quantify/requirements.txt

python quantify/quantize_jev_omni.py \
  --bits 8 \
  --source models/raw/jev-omni \
  --base-model models/raw/gemma-4-12B-it \
  --output models/jev-omni-int8 \
  --max-memory 0=28GiB \
  --max-memory 1=5GiB \
  --max-memory cpu=64GiB
```

The same command can be run from inside `quantify/`; paths beginning with `models/` still resolve under the project root. The first run downloads several large files and requires network access and enough disk space for both raw models and the quantized output.

The exporter writes standard PyTorch checkpoint shards directly from the quantized `state_dict`. This avoids a `Transformers 5.17` save bug involving bitsandbytes INT8 `SCB` metadata; seeing `model.save_pretrained` in a traceback means the Linux host is still running an older copy of the script.

Before each model load, the exporter checks real free VRAM and removes a GPU whose load-time headroom is too small. This prevents a busy secondary GPU from causing an OOM during Transformers' allocator warmup; the model is then placed on the remaining GPU and CPU offload.

For the smaller artifact:

```bash
python quantify/quantize_jev_omni.py \
  --bits 4 \
  --source models/raw/jev-omni \
  --base-model models/raw/gemma-4-12B-it \
  --output models/jev-omni-int4 \
  --max-memory 0=28GiB \
  --max-memory 1=5GiB \
  --max-memory cpu=64GiB
```

By default the Gemma base model is quantized with the same bit width. Use `--base-quantization none` only when preserving the base multimodal encoder is more important than memory usage.

Validate an artifact without loading the model:

```bash
python quantify/verify_quantized.py /opt/models/jev-omni-int8
```

`manifest.json` records the model IDs, quantization settings, runtime versions, and files included in the bundle. The artifact still requires a compatible CUDA, PyTorch, Transformers, and bitsandbytes runtime when it is served.
