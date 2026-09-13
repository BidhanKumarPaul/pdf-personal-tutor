"""
app.py
PDF -> Personal Tutor — Streamlit web app.

Upload lecture notes, ask questions, get answers grounded in your own
material with page citations. Deployable for free on Streamlit Community
Cloud (see README.md).
"""

import os

import streamlit as st

from src.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, TOP_K
from src.llm_client import LLMError, generate_answer
from src.pdf_processor import chunk_document
from src.vector_store import VectorStore, load_embedder

st.set_page_config(page_title="PDF Personal Tutor", page_icon="📚", layout="centered")


# ---------- Cached / session resources ----------

@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embedder():
    return load_embedder()


def get_api_key() -> str:
    """Checks Streamlit secrets first (for deployed app), then env var (for local dev)."""
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.environ.get("GROQ_API_KEY", "")


if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore(embedder=get_embedder())
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}


# ---------- Sidebar: upload & process documents ----------

with st.sidebar:
    st.header("📄 Your Notes")

    uploaded_files = st.file_uploader(
        "Upload PDF lecture notes",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs — they'll all be searchable together.",
    )

    if uploaded_files and st.button("Process documents", type="primary", use_container_width=True):
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

        if not new_files:
            st.info("These files are already processed.")
        else:
            progress = st.progress(0.0, text="Starting...")
            for i, file in enumerate(new_files):
                progress.progress(
                    (i) / len(new_files), text=f"Reading {file.name}..."
                )
                chunks = chunk_document(
                    pdf_source=file.getvalue(),
                    source_name=file.name,
                    chunk_size=CHUNK_SIZE_WORDS,
                    overlap=CHUNK_OVERLAP_WORDS,
                )
                if not chunks:
                    st.warning(f"No extractable text found in {file.name} (scanned image PDF?).")
                    continue

                st.session_state.vector_store.add(chunks)
                st.session_state.processed_files.append(file.name)

            progress.progress(1.0, text="Done!")
            st.success(f"Processed {len(new_files)} file(s).")
            st.rerun()

    if st.session_state.processed_files:
        st.divider()
        st.caption("Indexed documents:")
        for name in st.session_state.processed_files:
            st.caption(f"✅ {name}")

    st.divider()
    if not get_api_key():
        st.warning(
            "No Groq API key found. Add `GROQ_API_KEY` in your Streamlit secrets "
            "(deployed) or as an environment variable (local).",
            icon="⚠️",
        )
    else:
        st.caption("🔑 Groq API key loaded.")


# ---------- Main: chat interface ----------

st.title("📚 PDF → Personal Tutor")
st.caption("Upload your notes on the left, then ask anything about them.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your notes...")

if question:
    if not st.session_state.vector_store.is_ready:
        st.chat_message("assistant").warning(
            "Please upload and process at least one PDF first."
        )
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                matched_chunks = st.session_state.vector_store.search(question, k=TOP_K)
                try:
                    answer = generate_answer(question, matched_chunks, get_api_key())
                except LLMError as e:
                    answer = f"⚠️ {e}"

            st.markdown(answer)

            if matched_chunks:
                with st.expander("📎 Sources used"):
                    for c in matched_chunks:
                        st.caption(f"**{c.source}** — page {c.page}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
