# Changelog

## v2.0.0

- Replaced Chainlit with a Streamlit chat app.
- Renamed the app entry point from `SAMCares.py` to `app.py`.
- Reorganized `main_codes/` into the `samcares/` package.
- Replaced hardcoded Llama 2 loading with configurable Hugging Face model
  loading. The default is `Qwen/Qwen3-8B`.
- Moved Hugging Face authentication to the `HF_TOKEN` environment variable.
- Updated LangChain usage to current split packages and retrieval-chain helpers.
- Added Streamlit upload support for `.txt` and `.pdf` documents.
- Added CPU fallback for embeddings.
- Added `pyproject.toml`, `requirements.txt`, and `.env.example`.

