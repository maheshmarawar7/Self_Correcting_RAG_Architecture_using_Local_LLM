"""
ingest.py
---------
Ye script tumhare documents (PDF / TXT) ko:
  1. Load karta hai
  2. Chunks mein todta hai
  3. nomic-embed-text (Ollama) se embed karta hai
  4. Chroma vector DB mein store karta hai

Run: python ingest.py
Isse pehle "data/" folder mein apne PDFs/TXT files daal do.
"""

import os

# Chroma telemetry warnings band karne ke liye (import se PEHLE set karna zaroori hai)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")

import chromadb

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # .env se config (EMBED_MODEL, PERSIST_DIR, etc.)

DATA_DIR = os.getenv("DATA_DIR", "data")
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")   # ollama pull nomic-embed-text
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

_chroma_settings = chromadb.Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=PERSIST_DIR,
)


def resolve_data_dir():
    """Support both lowercase and uppercase folder names used across environments."""
    for candidate in [DATA_DIR, DATA_DIR.capitalize(), DATA_DIR.upper()]:
        if os.path.isdir(candidate):
            return candidate
    return DATA_DIR


def load_documents(data_dir: str):
    """Loads every .pdf and .txt file from the data/ folder."""
    docs = []

    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs.extend(pdf_loader.load())

    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    docs.extend(txt_loader.load())

    print(f"Loaded {len(docs)} raw documents from '{data_dir}'")
    return docs


def chunk_documents(docs):
    """Splits documents into overlapping chunks so retrieval is more precise."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """Embeds chunks and stores them in a persistent Chroma DB."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="local_knowledge_base",
        client_settings=_chroma_settings,
    )
    print(f"Vector store built and saved at '{PERSIST_DIR}'")
    return vectorstore


if __name__ == "__main__":
    resolved_data_dir = resolve_data_dir()
    if not os.path.isdir(resolved_data_dir) or not os.listdir(resolved_data_dir):
        print(f"'{resolved_data_dir}/' is empty. Put your PDFs/TXT files there first.")
        exit(1)

    documents = load_documents(resolved_data_dir)
    chunks = chunk_documents(documents)
    build_vectorstore(chunks)
    print("Ingestion complete. Ab main.py chala sakte ho.")
