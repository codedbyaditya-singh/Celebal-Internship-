import os
import re

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from langchain_core.documents import Document
from fpdf import FPDF


def load_documents(file_paths, pasted_text=None):
    documents = []

    # Load uploaded files
    for file_path in file_paths:

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            loader = PyPDFLoader(file_path)

        elif extension == ".txt":
            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

        else:
            continue

        documents.extend(loader.load())

    # Load manually entered notes/articles
    if pasted_text and pasted_text.strip():

        documents.append(
            Document(
                page_content=pasted_text,
                metadata={
                    "source": "Manual Text"
                }
            )
        )

    return documents

# PDF Export (Q&A / MCQ results)
def _safe(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _break_long_tokens(text, max_len=40):
    def _split(match):
        token = match.group(0)
        return " ".join(
            token[i:i + max_len] for i in range(0, len(token), max_len)
        )

    return re.sub(r"\S{%d,}" % (max_len + 1), _split, text)


def _write(pdf, text, line_height):
    safe_text = _break_long_tokens(_safe(text))
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, line_height, safe_text)


def generate_qa_pdf(title, items, kind="qa"):
    # Build a downloadable PDF of generated Q&A pairs or MCQs.
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    _write(pdf, title, 10)
    pdf.ln(4)

    for idx, item in enumerate(items, start=1):
        question = item.get("question", "")

        pdf.set_font("Helvetica", "B", 12)
        _write(pdf, f"Q{idx}. {question}", 8)

        if kind == "mcq":
            options = item.get("options", {}) or {}
            pdf.set_font("Helvetica", "", 11)
            for key in sorted(options.keys()):
                _write(pdf, f"   {key}) {options[key]}", 7)

            correct = item.get("correct_option", "")
            pdf.set_font("Helvetica", "B", 11)
            _write(pdf, f"   Correct Answer: {correct}", 7)

            explanation = item.get("explanation", "")
            if explanation:
                pdf.set_font("Helvetica", "I", 10)
                _write(pdf, f"   Explanation: {explanation}", 7)
        else:
            answer = item.get("answer", "")
            pdf.set_font("Helvetica", "", 11)
            _write(pdf, f"   Answer: {answer}", 7)

        pdf.ln(3)

    # fpdf2's output(dest="S") returns a bytearray-like object
    return bytes(pdf.output(dest="S"))