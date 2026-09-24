# Jev-Omni quantization

This directory creates complete, reusable Jev-Omni bundles. The source model and Gemma base model are copied into the output, while the decoder backbone and optionally the base model are saved with bitsandbytes INT8 or NF4 INT4 weights.

Run this on the Linux CUDA host that will perform the conversion. Relative model and output paths are resolved from the project root, so the commands work both from the repository root and from inside `quantify/`. The output directory must be new; the command never overwrites an existing artifact.

```bash
python -m pip install -r quantify/requirements.txt

python quantify/quantize_jev_omni.py \
  --bits 8 \
  --output /opt/models/jev-omni-int8 \
  --max-memory 0=28GiB \
  --max-memory 1=5GiB \
  --max-memory cpu=64GiB
```

For the smaller artifact:

```bash
python quantify/quantize_jev_omni.py \
  --bits 4 \
  --output /opt/models/jev-omni-int4 \
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
