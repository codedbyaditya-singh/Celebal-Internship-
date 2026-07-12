import hashlib
import json
import re
import shutil
import os

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_DB_DIR,
    TOP_K,
    DEFAULT_NUM_QUESTIONS,
    DEFAULT_NUM_MCQS,
    MAX_WHOLE_DOC_CHUNKS,
    INGESTED_MANIFEST_FILE,
)

# Embedding Model
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

# Gemini LLM
def get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )


def _extract_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)

    # Fallback: whatever it is, stringify it rather than crash
    return str(content)

# Chunking
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    return chunks

# Vector Store
@st.cache_resource
def _get_chroma_client():
    # A single, cached Chroma client for the lifetime of the app process.
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=get_embedding_model(),
    )


def create_vector_store(chunks):
    """
    Create a fresh Chroma vector database from scratch.
    If an old database exists, remove it first.
    """
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)

    # The old client (if any) now points at a deleted directory -
    # drop it so the next call opens a fresh connection.
    _get_chroma_client.clear()

    vector_store = _get_chroma_client()
    vector_store.add_documents(chunks)
    return vector_store


def load_vector_store():
    # Return the cached Chroma vector database connection.
    return _get_chroma_client()


def knowledge_base_exists():
    # Check whether the vector database exists.
    return os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0


def add_to_vector_store(chunks):
    if not chunks:
        return None

    if knowledge_base_exists():
        vector_store = load_vector_store()
        vector_store.add_documents(chunks)
        return vector_store

    return create_vector_store(chunks)

def reset_knowledge_base():
    """
    Fully wipe the vector store AND the ingestion manifest, so the
    next build starts completely from scratch.
    """
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)
    if os.path.exists(INGESTED_MANIFEST_FILE):
        os.remove(INGESTED_MANIFEST_FILE)

    # Drop the cached client - it's pointing at a directory that no
    # longer exists (or is about to be replaced).
    _get_chroma_client.clear()

# Ingestion Dedup (content-hash manifest)
def _compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _load_manifest():
    if not os.path.exists(INGESTED_MANIFEST_FILE):
        return {}
    try:
        with open(INGESTED_MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(manifest):
    with open(INGESTED_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def filter_new_sources(file_paths, manual_text=None):
    # If the vector store itself doesn't exist, any old manifest is.stale (points at data that no longer exists) - clear it first.
    if not knowledge_base_exists() and os.path.exists(INGESTED_MANIFEST_FILE):
        os.remove(INGESTED_MANIFEST_FILE)

    manifest = _load_manifest()

    new_file_paths = []
    skipped_labels = []
    pending = {}

    for path in file_paths:
        with open(path, "rb") as f:
            data = f.read()

        content_hash = _compute_hash(data)
        label = os.path.basename(path)

        if content_hash in manifest:
            skipped_labels.append(label)
        else:
            new_file_paths.append(path)
            pending[content_hash] = label

    new_manual_text = None
    if manual_text and manual_text.strip():
        text_hash = _compute_hash(manual_text.strip().encode("utf-8"))

        if text_hash in manifest:
            skipped_labels.append("Pasted text (duplicate)")
        else:
            new_manual_text = manual_text
            pending[text_hash] = "Manual Text"

    return new_file_paths, new_manual_text, skipped_labels, pending


def commit_ingested(pending):
    if not pending:
        return
    manifest = _load_manifest()
    manifest.update(pending)
    _save_manifest(manifest)

# Retrieval
def get_retriever(k=None):
    # Create a retriever from the vector database.
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": k or TOP_K
        }
    )
    return retriever


def retrieve_context(query, k=None):
    retriever = get_retriever(k=k)
    documents = retriever.invoke(query)
    return documents


def get_all_chunks(max_chunks=None):
    vector_store = load_vector_store()
    data = vector_store.get(include=["documents", "metadatas"])

    texts = data.get("documents", []) or []
    metadatas = data.get("metadatas", []) or []

    if max_chunks:
        texts = texts[:max_chunks]
        metadatas = metadatas[:max_chunks]

    documents = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(texts, metadatas)
    ]
    return documents

# JSON parsing helper
def _parse_json_array(raw_text):
    text = raw_text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            return json.loads(snippet)
        raise

# Generate Response (Core RAG Function)
def generate_response(query, task="qa"):
    llm = get_llm()
    documents = retrieve_context(query)
    context = "\n\n".join(
        [doc.page_content for doc in documents]
    )

    if task == "qa":
        prompt = f"""
You are a helpful AI assistant.
Answer ONLY using the provided context.
If the answer is not present in the context, reply:
"I couldn't find that information in the uploaded documents."
Context:
{context}
Question:
{query}
"""
    elif task == "summary":
        prompt = f"""
You are an expert summarizer.
Summarize the following document in clear bullet points.
Context:
{context}
"""
    elif task == "questions":
        prompt = f"""
Generate 10 important questions from the following content.
The questions should cover the major concepts.
Context:
{context}
"""
    else:
        raise ValueError("Invalid task.")

    try:
        response = llm.invoke(prompt)
        return _extract_text(response), documents
    except Exception as e:
        return f"Error: {str(e)}", []


# Important Questions + Answers
def generate_qa_pairs(topic=None, num_questions=DEFAULT_NUM_QUESTIONS):
    llm = get_llm()

    if topic and topic.strip():
        documents = retrieve_context(topic, k=max(TOP_K, 6))
    else:
        documents = get_all_chunks(max_chunks=MAX_WHOLE_DOC_CHUNKS)

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = f"""
You are an expert exam question setter.
Based ONLY on the context below, generate {num_questions} important
questions that test understanding of the key concepts, along with a
clear, accurate answer for each, grounded strictly in the context.

Return ONLY a valid JSON array with no extra text, no commentary, and
no markdown code fences. Each element must follow exactly this shape:
{{"question": "...", "answer": "..."}}

Context:
{context}
"""

    try:
        response = llm.invoke(prompt)
        qa_pairs = _parse_json_array(_extract_text(response))
        return qa_pairs, documents
    except Exception as e:
        return (
            [{"question": "Error generating questions", "answer": str(e)}],
            documents,
        )

# MCQ Generation
def generate_mcqs(topic=None, num_questions=DEFAULT_NUM_MCQS):
    llm = get_llm()

    if topic and topic.strip():
        documents = retrieve_context(topic, k=max(TOP_K, 6))
    else:
        documents = get_all_chunks(max_chunks=MAX_WHOLE_DOC_CHUNKS)

    context = "\n\n".join(doc.page_content for doc in documents)

    prompt = f"""
You are an expert quiz maker.
Based ONLY on the context below, generate {num_questions} multiple
choice questions (MCQs) that test understanding of the key concepts.
Each question must have exactly 4 options, only one of which is correct.

Return ONLY a valid JSON array with no extra text, no commentary, and
no markdown code fences. Each element must follow exactly this shape:
{{
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct_option": "A",
  "explanation": "..."
}}

Context:
{context}
"""

    try:
        response = llm.invoke(prompt)
        mcqs = _parse_json_array(_extract_text(response))
        return mcqs, documents
    except Exception as e:
        return (
            [{
                "question": "Error generating MCQs",
                "options": {},
                "correct_option": "",
                "explanation": str(e),
            }],
            documents,
        )