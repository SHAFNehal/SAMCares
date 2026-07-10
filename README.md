# SAMCares

SAMCares is a local study assistant for course documents. It lets a user upload
a `.txt` or `.pdf` file, builds a local FAISS vector database from that file,
and answers questions using retrieved document context.

The app is designed for local, single-user use through Streamlit.

Original project paper: https://arxiv.org/abs/2405.00330

## Features

- Streamlit chat interface.
- Upload support for `.txt` and `.pdf` study material.
- Local FAISS vector store for retrieval.
- Hugging Face text-generation model support.
- Configurable model, embedding model, retrieval count, and generation settings.
- CPU fallback for embeddings when CUDA is unavailable.
- Secrets loaded from environment variables instead of source code.

## Project Layout

```text
SAMCares/
  app.py                     # Streamlit app entry point
  samcares/
    ingestion.py             # Document loading, chunking, embeddings, FAISS creation
    rag.py                   # Model loading and RAG chain construction
    settings.py              # Environment-based runtime settings
  data/
    test_text_1.txt          # Sample text document
  public/
    logo_dark.png
    logo_light.png
  pyproject.toml             # uv project dependencies
  requirements.txt           # pip-compatible dependency fallback
  .env.example               # Example local configuration
```

Older files were reorganized:

| Old | New |
| --- | --- |
| `SAMCares.py` | `app.py` |
| `main_codes/database_preparation.py` | `samcares/ingestion.py` |
| `main_codes/llama2_model_text_generator.py` | `samcares/rag.py` |
| Chainlit app flow | Streamlit app flow |

## Requirements

- Python 3.11 or newer.
- `uv` for the recommended setup.
- Internet access the first time dependencies and Hugging Face models are
  downloaded.
- A Hugging Face token if the configured model is gated or private.
- GPU recommended for local LLM inference. CPU can work for embeddings, but the
  default generation model is large.

## Quick Start

From the project directory:

```bash
cd /Users/shafnehal/SAMCares
uv sync
uv run streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Hugging Face Token

The source code no longer contains a Hugging Face token. If your selected model
requires authentication, set `HF_TOKEN`.

Option 1: export it in your shell:

```bash
export HF_TOKEN="your_hugging_face_token"
```

Option 2: create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```text
HF_TOKEN=your_hugging_face_token
```

Do not commit `.env`.

## How To Use The App

1. Start the app with `uv run streamlit run app.py`.
2. In the sidebar, upload a `.txt` or `.pdf` file.
3. Click `Build Vector Store`.
4. Wait for the success message showing how many chunks were indexed.
5. Ask questions in the chat input.
6. Use `Clear Chat` to reset the conversation while keeping the vector store.

The vector store is saved to `vector_data/` by default. It can be reused across
app restarts.

## Configuration

SAMCares is configured with environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | unset | Hugging Face token for gated/private models |
| `SAMCARES_MODEL_ID` | `Qwen/Qwen3-8B` | Hugging Face text-generation model |
| `SAMCARES_EMBEDDING_MODEL_ID` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model used for FAISS |
| `SAMCARES_VECTOR_STORE_PATH` | `./vector_data` | Where the FAISS index is saved |
| `SAMCARES_MAX_NEW_TOKENS` | `1024` | Maximum generated tokens per answer |
| `SAMCARES_TEMPERATURE` | `0.7` | Sampling temperature |
| `SAMCARES_TOP_P` | `0.8` | Nucleus sampling value |
| `SAMCARES_TOP_K` | `20` | Top-k sampling value |
| `SAMCARES_RETRIEVER_K` | `6` | Number of retrieved chunks per question |

Example:

```bash
export SAMCARES_MODEL_ID="Qwen/Qwen3-4B"
export SAMCARES_MAX_NEW_TOKENS="512"
uv run streamlit run app.py
```

## Supported Documents

Supported upload types:

- `.txt`
- `.pdf`

Notes:

- PDF text extraction depends on the PDF containing selectable text.
- Scanned image-only PDFs need OCR first.
- Large files can take time to chunk and embed.
- The current app builds one active vector store at a time.

## Dependency Management

Recommended:

```bash
uv sync
```

Fallback with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The project intentionally uses LangChain `0.3.x` APIs. Do not upgrade to
LangChain `1.x` without updating `samcares/rag.py`, because the chain helper
imports changed in LangChain `1.x`.

## Verification

Useful checks after setup or code changes:

```bash
uv run python -m compileall app.py samcares
uv run python -c "import app; import samcares.ingestion; import samcares.rag; print('imports ok')"
uv run python -c "from samcares.ingestion import load_documents, split_documents; docs=load_documents('data/test_text_1.txt'); print(len(split_documents(docs)))"
```

You can also build a vector store from the sample document:

```bash
uv run python -c "from samcares.ingestion import create_vector_store_from_path; print(create_vector_store_from_path('data/test_text_1.txt'))"
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'streamlit'`

Run:

```bash
uv sync
```

Then start the app with:

```bash
uv run streamlit run app.py
```

### Hugging Face model access error

Set `HF_TOKEN` and confirm your Hugging Face account has access to the selected
model.

### The app is slow or runs out of memory

The default model, `Qwen/Qwen3-8B`, is large. Use a smaller compatible model by
setting `SAMCARES_MODEL_ID`, reduce `SAMCARES_MAX_NEW_TOKENS`, or run on a
machine with more GPU memory.

### PDF upload succeeds but answers are poor

The PDF may not contain extractable text, or the extracted text may be noisy.
Try converting the document to clean text first.

### Existing vector store will not load

Delete `vector_data/` and rebuild it from the app. Only load FAISS indexes that
you created or otherwise trust.

## Security Notes

- Do not commit `.env` or real tokens.
- `vector_data/` is a local runtime artifact and is ignored by Git.
- FAISS loading uses local deserialization. Treat vector stores as trusted local
  files only.

## Legacy Notes

`environment.yml` is kept for historical reproducibility from the original
prototype. New development should use `pyproject.toml`, `uv.lock`, and `uv`.
