# rag.py - Retrieval-Augmented Generation over user-uploaded documents.
#
# Documents are chunked, embedded with Gemini, and stored in a local Chroma vector store.
# Every chunk carries the owner's username in its metadata, and every retrieval/deletion is
# filtered by username -- so users can only ever retrieve their own documents (the same
# isolation principle as the SQL table allowlist).
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_store")
# Embedding model — overridable via env. Must be one your API key supports
# (list with: GET https://generativelanguage.googleapis.com/v1beta/models).
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

_client = None
_collection = None
_embeddings = None

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _emb():
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    return _embeddings


def _collection_handle():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection("documents")
    return _collection


def ingest_document(username: str, doc_name: str, text: str) -> int:
    """Chunk, embed, and store a document's text. Returns the number of chunks stored."""
    chunks = [c for c in _splitter.split_text(text) if c.strip()]
    if not chunks:
        return 0
    embeddings = _emb().embed_documents(chunks)
    ids = [f"{username}::{doc_name}::{i}" for i in range(len(chunks))]
    metadatas = [{"username": username, "doc_name": doc_name, "chunk_id": i} for i in range(len(chunks))]
    _collection_handle().add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def query_documents(username: str, question: str, k: int = 4):
    """Retrieve the top-k chunks from the user's own documents. Returns [(text, metadata), ...]."""
    q_emb = _emb().embed_query(question)
    res = _collection_handle().query(
        query_embeddings=[q_emb],
        n_results=k,
        where={"username": username},
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    return list(zip(docs, metas))


def delete_document(username: str, doc_name: str) -> None:
    """Remove all chunks of a specific document owned by the user."""
    _collection_handle().delete(
        where={"$and": [{"username": username}, {"doc_name": doc_name}]}
    )
