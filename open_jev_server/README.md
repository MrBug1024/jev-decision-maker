# Original open-jev server snapshot

This directory is a frozen copy of the existing service for `open-jev-deberta-v3-large`. It is intentionally independent from `jev_omni_server` and `quantify`.

Start it with the migrated `.env` and `python server.py`. Its default database is `../data/jev_gateway.db`, shared with `jev_omni_server`, so accounts, sessions, and API Keys are valid in both services.

The copied `.cache` directory contains the existing Hugging Face cache. Keep the `.env` file local to this deployment and do not commit it.