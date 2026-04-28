# Plan: SAMCares Modernization

## Goal

Modernize SAMCares from a Chainlit + Llama 2 prototype into a Streamlit-based RAG app with safer configuration, current LangChain APIs, explicit dependency management, and a cleaner document ingestion flow.

The first implementation target is a local single-user app that can:

1. Accept `.txt` and `.pdf` study material from the UI.
2. Build or refresh a local FAISS vector store.
3. Answer questions with retrieved context.
4. Preserve chat history during a Streamlit session.
5. Run through `uv` with documented setup steps.

## Current State

| Area | Current Implementation | Issue |
| --- | --- | --- |
| UI | Chainlit callbacks in `SAMCares.py` | Hard to combine with upload-driven setup and blocks on `input()` at import time |
| Configuration | `login("hf_YOUR_HUGGINGFACE_TOKEN")` | Token handling belongs in environment variables |
| LLM | `meta-llama/Llama-2-70b-chat-hf` through Transformers | Very large default model and gated access make local setup difficult |
| RAG chain | `RetrievalQA.from_chain_type()` with `ConversationBufferMemory` | Deprecated LangChain pattern |
| Vector DB | CLI prompts load one file from `./data` | No web upload flow; assumes CUDA for embeddings |
| Dependencies | Large Conda `environment.yml` | Difficult to reproduce quickly; includes many transitive and environment-specific pins |

## Target Architecture

```text
Streamlit UI
  - sidebar document upload
  - chat display and chat input
  - session_state for messages, vector-store status, and chain

Document ingestion
  - save uploaded file to a temporary path
  - load PDF/TXT through LangChain community loaders
  - split documents
  - embed chunks
  - save FAISS index locally

RAG pipeline
  - FAISS retriever
  - ChatPromptTemplate with system prompt, context, and history
  - create_stuff_documents_chain
  - create_retrieval_chain
  - explicit chat_history passed from Streamlit

Model layer
  - default model selected by configuration
  - Hugging Face token read from HF_TOKEN when needed
  - CPU fallback for embeddings when CUDA is unavailable
```

## Implementation Phases

### Phase 1: Project Hygiene And Configuration

1. Add `pyproject.toml` as the primary project definition for `uv`.
2. Add `requirements.txt` as a simple compatibility install path.
3. Keep `environment.yml` for historical reproducibility, but mark it as legacy in the README.
4. Update `.gitignore` for local runtime artifacts:
   - `.venv/`
   - `__pycache__/`
   - `.env`
   - `vector_data/`
   - `*.gguf`
   - uploaded PDFs or large local documents
5. Move Hugging Face authentication to `HF_TOKEN`.
6. Add `.env.example` with non-secret placeholders.

Acceptance criteria:

1. `uv sync` creates a working environment.
2. No source file contains a real or placeholder token passed directly to `login()`.
3. Runtime-generated vector data and model files are ignored.

### Phase 2: Streamlit App Shell

Rewrite `SAMCares.py` from Chainlit callbacks to Streamlit:

1. Replace `chainlit` imports with `streamlit`.
2. Remove all module-level `input()` calls.
3. Add `st.set_page_config()`.
4. Add sidebar controls:
   - document uploader for `.txt` and `.pdf`
   - rebuild vector database button
   - clear chat button
   - status for active vector store
5. Store app state in `st.session_state`:
   - `messages`
   - `chain`
   - `vector_db_ready`
6. Render prior messages with `st.chat_message()`.
7. Accept new messages with `st.chat_input()`.
8. Convert stored messages into LangChain `HumanMessage` and `AIMessage` objects before invoking the chain.

Acceptance criteria:

1. `uv run streamlit run SAMCares.py` starts without blocking for terminal input.
2. The app can clear and replay chat state correctly across reruns.
3. The user sees a clear message when no vector database is available.

### Phase 3: Document Ingestion

Update `main_codes/database_preparation.py`:

1. Replace deprecated imports with split LangChain packages:
   - `langchain_community.vectorstores.FAISS`
   - `langchain_huggingface.HuggingFaceEmbeddings`
   - `langchain_text_splitters.CharacterTextSplitter` or `RecursiveCharacterTextSplitter`
   - `langchain_community.document_loaders.PyPDFLoader`
   - `langchain_community.document_loaders.TextLoader`
2. Remove unused imports.
3. Keep the existing CLI function for backward compatibility.
4. Add a Streamlit-friendly function:
   - `create_vector_database_from_upload(uploaded_file, vector_database_path="./vector_data")`
5. Use `tempfile.NamedTemporaryFile` for uploaded files.
6. Detect file type from suffix and reject unsupported files with a clear exception.
7. Add an embedding-device helper:
   - use CUDA when available
   - otherwise use CPU

Acceptance criteria:

1. TXT upload creates a FAISS index.
2. PDF upload creates a FAISS index when `pypdf` is installed.
3. Unsupported file types fail with a readable error.
4. CPU-only machines can still build embeddings.

### Phase 4: RAG Chain And Model Layer

Update `main_codes/llama2_model_text_generator.py`:

1. Replace deprecated LangChain imports with current split packages.
2. Replace `RetrievalQA.from_chain_type()` with LCEL helpers:
   - `create_stuff_documents_chain`
   - `create_retrieval_chain`
3. Remove `ConversationBufferMemory`; pass `chat_history` explicitly.
4. Replace the Llama 2 `[INST]` prompt format with `ChatPromptTemplate`.
5. Keep the SAMCares study-buddy persona, but make the prompt shorter and easier to audit.
6. Make model selection configurable:
   - `SAMCARES_MODEL_ID`, defaulting to a practical Hugging Face chat model selected during implementation
   - `SAMCARES_MODEL_MODE`, such as `transformers` or `llama_cpp`
7. Verify model names and recommended generation settings against current upstream docs before implementation.
8. Keep a lightweight local path optional rather than required.

Acceptance criteria:

1. The chain accepts `{"input": question, "chat_history": history}` or an equivalent documented schema.
2. Retrieval uses the local FAISS index.
3. Responses include document context when relevant.
4. The app fails clearly when model weights are unavailable or the HF token is missing.

### Phase 5: Documentation

Rewrite `README.md`:

1. Explain what SAMCares does.
2. Document the new setup:
   - install `uv`
   - `uv sync`
   - set `HF_TOKEN` only if the configured model requires it
   - `uv run streamlit run SAMCares.py`
3. Document supported uploads and vector store behavior.
4. Explain CPU vs GPU expectations.
5. Preserve the original paper link.
6. Mention that `environment.yml` is legacy.

Add `CHANGELOG.md`:

1. Create a `v2.0.0` entry for the modernization.
2. List breaking changes:
   - Chainlit replaced by Streamlit
   - old CLI startup flow removed from the main app
   - Llama 2 no longer hardcoded

Acceptance criteria:

1. A new developer can run the app from the README alone.
2. Secrets are documented without being committed.
3. Model and hardware requirements are explicit.

## Proposed File Changes

| File | Planned Change |
| --- | --- |
| `SAMCares.py` | Full Streamlit rewrite |
| `main_codes/database_preparation.py` | Updated imports, upload ingestion, CPU/GPU embedding helper |
| `main_codes/llama2_model_text_generator.py` | Current LangChain chain, configurable model loading, prompt rewrite |
| `README.md` | Full setup and usage rewrite |
| `.gitignore` | Ignore runtime, secret, and model artifacts |
| `pyproject.toml` | New `uv` project metadata |
| `requirements.txt` | Flat dependency fallback |
| `.env.example` | New environment variable template |
| `CHANGELOG.md` | New modernization changelog |

## Dependency Targets

Use broad minimums where possible, then let `uv.lock` capture exact resolution:

```toml
[project]
name = "samcares"
version = "2.0.0"
requires-python = ">=3.11"
dependencies = [
  "streamlit>=1.36",
  "langchain>=0.3",
  "langchain-community>=0.3",
  "langchain-core>=0.3",
  "langchain-huggingface>=0.1",
  "langchain-text-splitters>=0.3",
  "transformers>=4.51",
  "torch",
  "faiss-cpu",
  "sentence-transformers",
  "huggingface-hub",
  "pypdf",
  "python-dotenv",
]
```

Optional local GGUF support can be added as an extra:

```toml
[project.optional-dependencies]
llama-cpp = ["llama-cpp-python"]
```

## Verification Checklist

Run these before considering the migration complete:

1. `uv sync`
2. `uv run python -m compileall SAMCares.py main_codes`
3. `uv run streamlit run SAMCares.py`
4. Upload `data/test_text_1.txt` and build the vector database.
5. Ask at least two questions that require retrieved context.
6. Clear chat and confirm the vector database remains available.
7. Restart the app and confirm an existing FAISS index can load.
8. Run once on CPU-only mode or force CPU embeddings to verify fallback.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Model availability or recommended IDs change | Verify selected model and generation settings immediately before implementation |
| Local machines lack enough VRAM | Make model ID configurable and document CPU/GPU expectations |
| FAISS deserialization can be unsafe | Only load locally generated indexes; document that indexes must be trusted |
| LangChain APIs continue shifting | Keep imports isolated in the model and ingestion modules |
| Streamlit reruns can rebuild expensive resources | Cache chain/vector loading carefully and rebuild only on explicit upload action |
| Large uploaded files can stall the app | Add clear size guidance and later consider background processing |

## Out Of Scope For First Migration

1. Multi-user authentication.
2. Persistent per-user document collections.
3. Database-backed chat history.
4. Cloud deployment.
5. Automated evaluation of answer quality.
6. Support for file types beyond `.txt` and `.pdf`.

## Open Decisions

1. Which model should be the default for typical local hardware?
2. Should the project keep a CLI ingestion command in addition to Streamlit upload?
3. Should generated `uv.lock` be committed?
4. What maximum upload size should be supported?
5. Should source citations be shown in chat responses?
