import os
import streamlit as st

from utils import load_documents, generate_qa_pdf

from rag_pipeline import (
    split_documents,
    add_to_vector_store,
    generate_response,
    generate_qa_pairs,
    generate_mcqs,
    knowledge_base_exists,
    filter_new_sources,
    commit_ingested,
    reset_knowledge_base,
)

from config import DEFAULT_NUM_QUESTIONS, DEFAULT_NUM_MCQS

# Page Configuration
st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Retrieval-Augmented Generation (RAG)")
st.write("Upload your documents and ask questions based on their content.")

# Session State s
if "last_result" not in st.session_state:
    st.session_state.last_result = None  

# Sidebar - Build Knowledge Base
st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

manual_text = st.sidebar.text_area(
    "Paste Notes / Articles (Optional)"
)

task = st.sidebar.selectbox(
    "Select Task",
    (
        "Question Answering",
        "Summarize Document",
        "Generate Important Questions (with Answers)",
        "Generate MCQs",
    ),
)

col_build, col_reset = st.sidebar.columns(2)

build_clicked = col_build.button("Build Knowledge Base")
reset_clicked = col_reset.button("Reset")

if reset_clicked:
    reset_knowledge_base()
    st.session_state.last_result = None
    st.sidebar.success("Knowledge base cleared. Upload and build again.")

if build_clicked:

    if not uploaded_files and not manual_text.strip():
        st.warning("Please upload documents or paste some text.")

    else:
        os.makedirs("uploads", exist_ok=True)

        file_paths = []

        for file in uploaded_files:
            file_path = os.path.join("uploads", file.name)

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            file_paths.append(file_path)

# checking same pdf so no reupload n re chunk/re embeding happens
        new_file_paths, new_manual_text, skipped_labels, pending = filter_new_sources(
            file_paths, manual_text
        )

        if not new_file_paths and not new_manual_text:
            st.info(
                "This content is already in the knowledge base — "
                "nothing new to add."
            )

        else:
            with st.spinner("Building Knowledge Base..."):

                documents = load_documents(new_file_paths, new_manual_text)

                if not documents:
                    st.error("No valid text found.")
                    st.stop()

                chunks = split_documents(documents)
                add_to_vector_store(chunks)
                commit_ingested(pending)

            st.session_state.last_result = None
            st.success("Knowledge Base Updated Successfully!")

            if skipped_labels:
                st.info(
                    "Skipped (already in knowledge base): "
                    + ", ".join(skipped_labels)
                )

# Main Panel
st.divider()

is_generation_task = task in (
    "Generate Important Questions (with Answers)",
    "Generate MCQs",
)

query = ""
scope_topic = ""
num_items = DEFAULT_NUM_QUESTIONS

if is_generation_task:

    st.subheader("Generation Settings")

    scope = st.radio(
        "Scope",
        ("Whole Document", "Specific Topic / Query"),
        horizontal=True,
    )

    if scope == "Specific Topic / Query":
        scope_topic = st.text_input(
            "Enter the topic or query to focus on"
        )

    default_count = (
        DEFAULT_NUM_QUESTIONS
        if task == "Generate Important Questions (with Answers)"
        else DEFAULT_NUM_MCQS
    )

    num_items = st.number_input(
        "How many to generate?",
        min_value=1,
        max_value=25,
        value=default_count,
        step=1,
    )

    generate_clicked = st.button("Generate")

else:
    st.subheader("Ask Your Query")

    query = st.text_area("Enter your question or request")

    generate_clicked = st.button("Generate Response")

# Handle Generation
if generate_clicked:

    if not knowledge_base_exists():
        st.error("Please build the Knowledge Base first.")

    elif not is_generation_task and not query.strip():
        st.warning("Please enter a query.")

    elif (
        is_generation_task
        and scope == "Specific Topic / Query"
        and not scope_topic.strip()
    ):
        st.warning("Please enter a topic/query, or switch scope to 'Whole Document'.")

    else:
        with st.spinner("Generating response..."):

            if task == "Question Answering":
                response, documents = generate_response(query=query, task="qa")
                st.session_state.last_result = {
                    "type": "text",
                    "response": response,
                    "documents": documents,
                }

            elif task == "Summarize Document":
                response, documents = generate_response(query=query, task="summary")
                st.session_state.last_result = {
                    "type": "text",
                    "response": response,
                    "documents": documents,
                }

            elif task == "Generate Important Questions (with Answers)":
                topic = scope_topic if scope == "Specific Topic / Query" else None
                qa_pairs, documents = generate_qa_pairs(
                    topic=topic, num_questions=int(num_items)
                )
                st.session_state.last_result = {
                    "type": "qa",
                    "items": qa_pairs,
                    "documents": documents,
                }

            else:  # Generate MCQs
                topic = scope_topic if scope == "Specific Topic / Query" else None
                mcqs, documents = generate_mcqs(
                    topic=topic, num_questions=int(num_items)
                )
                st.session_state.last_result = {
                    "type": "mcq",
                    "items": mcqs,
                    "documents": documents,
                }

        st.success("Response Generated!")

# Display Results
result = st.session_state.last_result

if result:

    st.divider()
    st.subheader("Result")

    if result["type"] == "text":
        st.write(result["response"])

    elif result["type"] == "qa":
        for i, item in enumerate(result["items"], start=1):
            st.markdown(f"**Q{i}. {item.get('question', '')}**")
            with st.expander("Show Answer"):
                st.write(item.get("answer", ""))

        pdf_bytes = generate_qa_pdf(
            "Important Questions & Answers", result["items"], kind="qa"
        )
        st.download_button(
            label="⬇️ Download Q&A as PDF",
            data=pdf_bytes,
            file_name="important_questions_answers.pdf",
            mime="application/pdf",
        )

    elif result["type"] == "mcq":
        for i, item in enumerate(result["items"], start=1):
            st.markdown(f"**Q{i}. {item.get('question', '')}**")
            options = item.get("options", {}) or {}
            for key in sorted(options.keys()):
                st.write(f"{key}) {options[key]}")

            with st.expander("Show Correct Answer"):
                st.write(f"Correct Option: {item.get('correct_option', '')}")
                if item.get("explanation"):
                    st.caption(item.get("explanation"))

        pdf_bytes = generate_qa_pdf(
            "Multiple Choice Questions", result["items"], kind="mcq"
        )
        st.download_button(
            label="⬇️ Download MCQs as PDF",
            data=pdf_bytes,
            file_name="mcqs.pdf",
            mime="application/pdf",
        )

    st.divider()
    st.subheader("Retrieved Context")

    for i, doc in enumerate(result["documents"], start=1):
        with st.expander(f"Chunk {i}"):
            st.write(doc.page_content)
            st.caption(doc.metadata)