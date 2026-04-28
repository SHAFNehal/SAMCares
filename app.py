"""Streamlit entry point for SAMCares."""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from samcares.ingestion import create_vector_store_from_upload, embedding_device
from samcares.rag import build_rag_chain
from samcares.settings import settings


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chain", None)
    st.session_state.setdefault("vector_db_ready", settings.vector_store_path.exists())
    st.session_state.setdefault("active_document", None)


def to_langchain_history(messages: list[dict[str, str]]):
    history = []
    for message in messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))
    return history


def ensure_chain():
    if not st.session_state.vector_db_ready:
        return None
    if st.session_state.chain is None:
        st.session_state.chain = build_rag_chain(settings.vector_store_path)
    return st.session_state.chain


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Documents")
        uploaded_file = st.file_uploader(
            "Upload study material",
            type=["txt", "pdf"],
            accept_multiple_files=False,
        )

        if st.button("Build Vector Store", type="primary", disabled=uploaded_file is None):
            if uploaded_file is None:
                st.warning("Upload a TXT or PDF file first.")
            else:
                with st.spinner("Indexing document..."):
                    try:
                        chunk_count = create_vector_store_from_upload(
                            uploaded_file,
                            uploaded_file.name,
                            settings.vector_store_path,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.session_state.vector_db_ready = False
                        st.session_state.chain = None
                        st.error(f"Could not build vector store: {exc}")
                    else:
                        st.session_state.vector_db_ready = True
                        st.session_state.chain = None
                        st.session_state.active_document = uploaded_file.name
                        st.success(f"Indexed {chunk_count} chunks from {uploaded_file.name}.")

        if st.button("Clear Chat"):
            st.session_state.messages = []

        st.divider()
        st.caption(f"Model: `{settings.model_id}`")
        st.caption(f"Embeddings: `{settings.embedding_model_id}`")
        st.caption(f"Embedding device: `{embedding_device()}`")
        st.caption(f"Vector store: `{settings.vector_store_path}`")

        if st.session_state.vector_db_ready:
            st.success("Vector store is ready.")
        else:
            st.info("Upload a document to create a vector store.")


def render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    st.set_page_config(
        page_title="SAMCares",
        page_icon="public/logo_light.png",
        layout="wide",
    )
    init_state()
    render_sidebar()

    st.title("SAMCares")
    st.caption("A local RAG study buddy for course documents.")

    render_messages()

    question = st.chat_input("Ask a question about your study material")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not st.session_state.vector_db_ready:
        answer = "Upload a TXT or PDF document and build the vector store before asking questions."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.warning(answer)
        return

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = ensure_chain()
                result = chain.invoke(
                    {
                        "input": question,
                        "chat_history": to_langchain_history(
                            st.session_state.messages[:-1]
                        ),
                    }
                )
                answer = result.get("answer", "I could not generate an answer.")
            except Exception as exc:  # noqa: BLE001
                answer = f"Could not generate an answer: {exc}"
                st.error(answer)
            else:
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

