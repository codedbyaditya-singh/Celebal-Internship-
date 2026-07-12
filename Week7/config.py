import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-3.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

VECTOR_DB_DIR = "chroma_db"
TOP_K = 4

# Tracks a content-hash of every file/text already embedded, so
INGESTED_MANIFEST_FILE = "ingested_manifest.json"

# QUESTION / MCQ GENERATION
# Default number of items to generate
DEFAULT_NUM_QUESTIONS = 10
DEFAULT_NUM_MCQS = 5

# When generating from the "whole document" (no specific topic),cap how many stored chunks are pulled into the prompt so the context sent to the LLM doesn't blow up for large PDFs.
MAX_WHOLE_DOC_CHUNKS = 30