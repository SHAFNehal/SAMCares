"""RAG chain and model loading for SAMCares."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import login
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from samcares.ingestion import build_embeddings
from samcares.settings import settings


SYSTEM_PROMPT = """You are SAMCares, a friendly study buddy for students.
Answer using the supplied context and the conversation history.
Be concise, accurate, respectful, and honest.
If the context does not contain enough information, say what is missing instead of inventing an answer.

Context:
{context}"""


def authenticate_hugging_face() -> None:
    """Authenticate only when an HF_TOKEN is configured."""

    if settings.hf_token:
        login(token=settings.hf_token, add_to_git_credential=False)


def load_llm() -> HuggingFacePipeline:
    """Load the configured Hugging Face text-generation model."""

    authenticate_hugging_face()

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_id,
        token=settings.hf_token,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_id,
        token=settings.hf_token,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=settings.max_new_tokens,
        do_sample=True,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=generator)


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def load_vector_store(vector_store_path: str | Path = settings.vector_store_path) -> FAISS:
    path = Path(vector_store_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {path}. Upload a document first."
        )

    return FAISS.load_local(
        str(path),
        build_embeddings(),
        allow_dangerous_deserialization=True,
    )


def build_rag_chain(vector_store_path: str | Path = settings.vector_store_path):
    vector_store = load_vector_store(vector_store_path)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )
    document_chain = create_stuff_documents_chain(load_llm(), build_prompt())
    return create_retrieval_chain(retriever, document_chain)


def gpu_available() -> bool:
    return torch.cuda.is_available()

