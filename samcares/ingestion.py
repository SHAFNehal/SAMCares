"""Document loading and FAISS vector store creation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

import torch
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from samcares.settings import settings


SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def embedding_device() -> str:
    """Return the best available device for local embeddings."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_id,
        model_kwargs={"device": embedding_device()},
    )


def load_documents(file_path: str | Path) -> list[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: .pdf, .txt."
        )

    return loader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=160,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vector_store_from_path(
    file_path: str | Path,
    vector_store_path: str | Path = settings.vector_store_path,
) -> int:
    """Create and persist a FAISS vector store from a PDF or TXT file."""

    documents = load_documents(file_path)
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text could be extracted from the selected document.")

    vector_store = FAISS.from_documents(chunks, build_embeddings())
    vector_store.save_local(str(vector_store_path))
    return len(chunks)


def create_vector_store_from_upload(
    uploaded_file: BinaryIO,
    filename: str,
    vector_store_path: str | Path = settings.vector_store_path,
) -> int:
    """Create and persist a FAISS vector store from a Streamlit upload."""

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported types: .pdf, .txt."
        )

    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = Path(temp_file.name)

    try:
        return create_vector_store_from_path(temp_path, vector_store_path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_user_data(data_types: list[str], directory: str) -> list[Document]:
    """Backward-compatible CLI loader for local PDF/TXT files."""

    data_dir = Path(directory)
    if not data_dir.exists():
        raise ValueError(f"Directory not found: {directory}")

    print("\nAvailable files:")
    for file_name in os.listdir(data_dir):
        print(file_name)

    print("Available data types:", data_types)
    print("Currently only PDF and TXT files are supported.\n")
    data_type = input("Enter the data type: ").lower()

    if data_type not in data_types:
        raise ValueError(f"Invalid data type selected: {data_type}")

    file_name = input("\nEnter the name of the file to load: ") + f".{data_type}"
    file_path = data_dir / file_name

    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    return load_documents(file_path)


def create_vector_database(
    data_types: list[str] | None = None,
    directory: str = "./data",
    vector_database_path: str = "./vector_data",
) -> int:
    """Backward-compatible CLI vector database builder."""

    documents = load_user_data(data_types or ["pdf", "txt"], directory)
    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text could be extracted from the selected document.")

    vector_store = FAISS.from_documents(chunks, build_embeddings())
    vector_store.save_local(vector_database_path)

    print("#################################################################")
    print("# Completed creating vector database and saved in local folder! #")
    print("#################################################################")
    return len(chunks)
