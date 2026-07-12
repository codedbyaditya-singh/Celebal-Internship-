# Week 7 Assignment — RAG-Based Question Answering System

A Retrieval-Augmented Generation (RAG) system that answers questions, generates important Q&A pairs, and creates MCQs from custom documents (PDFs, text files, or pasted notes) — built as part of a Week 7 internship assignment.

## 🚀 Features

- **Document Ingestion** — upload PDFs, `.txt` files, or paste raw text/notes directly
- **Question Answering** — ask questions and get answers grounded strictly in the uploaded document's content
- **Document Summarization** — generate a clear, bullet-point summary of the uploaded content
- **Important Q&A Generation** — auto-generate important questions *with* answers, scoped to the whole document or a specific topic
- **MCQ Generation** — auto-generate multiple-choice questions (4 options + correct answer + explanation), scoped to the whole document or a specific topic
- **PDF Export** — download generated Q&A pairs or MCQs as a formatted PDF
- **Duplicate Detection** — content-hash based deduplication, so re-uploading the same file doesn't re-embed it
- **Persistent Vector Store** — knowledge base persists across sessions using ChromaDB

## 🛠️ Tech Stack

- **Language:** Python
- **UI Framework:** Streamlit
- **Orchestration:** LangChain
- **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database:** ChromaDB
- **LLM:** Google Gemini (`gemini-3.5-flash`) via `langchain-google-genai`
- **PDF Processing:** `pypdf`, `fpdf2`
- **Environment Management:** `python-dotenv`

## 📂 Project Structure

```
rag_qa_system/
├── app.py                 # Streamlit UI and main app flow
├── config.py               # Configuration (models, chunk size, paths)
├── rag_pipeline.py          # Core RAG logic: ingestion, embedding, retrieval, generation
├── utils.py                 # Document loading and PDF export utilities
├── requirements.txt          # Python dependencies
└── .env.example               # Template for required environment variables
```

## ⚙️ How It Works (Pipeline)

1. **Document Ingestion** — PDFs/text files are loaded and converted to raw text
2. **Text Chunking** — text is split into overlapping chunks using LangChain's recursive character splitter
3. **Embedding Creation** — each chunk is converted into a vector using the MiniLM embedding model
4. **Vector Storage** — embeddings are stored in ChromaDB for fast similarity search
5. **Query Processing** — user questions are embedded and matched against stored chunks
6. **Context Retrieval** — the most relevant chunks are retrieved
7. **Answer Generation** — Gemini generates a grounded response using the retrieved context

## 🧰 Setup Instructions

1. Clone the repository
2. Create a virtual environment and activate it
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your own API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📌 Notes

- Works best with plain-text-heavy PDFs (reports, articles, notes). Table/diagram-heavy slide decks may extract text unreliably due to PDF text-layer limitations.
- "Whole Document" question/MCQ generation samples a capped number of chunks for very large documents — use the "Specific Topic" scope for targeted coverage of later sections in large PDFs.

## 📄 License

This project was built for educational/internship assignment purposes.
