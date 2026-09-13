# 📚 PDF → Personal Tutor

Upload your lecture notes as a PDF. Ask questions in plain English. Get
answers grounded in *your* material, with page citations — powered by a
local embedding model (free) and Groq's LLM API (free tier).

**[Live demo →](#)** *(update this link once you deploy)*

---

## How it works

This is a Retrieval-Augmented Generation (RAG) app:

1. **Extract** — `pdfplumber` pulls text from your PDF, page by page.
2. **Chunk** — text is split into overlapping ~300-word chunks so context
   isn't lost across boundaries.
3. **Embed** — each chunk is converted to a vector locally using
   `sentence-transformers` (`all-MiniLM-L6-v2`). No API calls, no cost.
4. **Index** — vectors go into a FAISS index for fast similarity search.
5. **Retrieve** — when you ask a question, it's embedded too, and the
   most similar chunks are pulled from the index.
6. **Generate** — the question + matched chunks are sent to Groq's free
   LLM API, which writes the actual explanation.

```
PDF Upload → Extract Text → Chunk → Embed (local) → FAISS Index
                                                          ↓
Your Question → Embed (local) → Search Index → Top Chunks
                                                          ↓
                                    Groq LLM → Answer + Citations
```

## Project structure

```
pdf-personal-tutor/
├── app.py                      # Streamlit web app (entry point)
├── src/
│   ├── config.py                # All tunable constants
│   ├── pdf_processor.py         # PDF text extraction + chunking
│   ├── vector_store.py          # Embedding + FAISS wrapper
│   └── llm_client.py            # Groq API client
├── tests/
│   └── test_pdf_processor.py    # Unit tests for chunking
├── .streamlit/
│   └── secrets.toml.example     # Template — copy to secrets.toml locally
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Run it locally

```bash
git clone https://github.com/<your-username>/pdf-personal-tutor.git
cd pdf-personal-tutor
pip install -r requirements.txt

# Add your API key for local dev
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your key

streamlit run app.py
```

Get a **free** Groq API key at [console.groq.com](https://console.groq.com).

## Deploy it live (free)

**Streamlit Community Cloud** is the fastest path — free hosting, made by
the same team as the framework.

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repo, branch `main`, and file path `app.py`.
4. Before/after deploying, go to **App settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_actual_key_here"
   ```
5. Click **Deploy**. You'll get a public URL like
   `https://your-app-name.streamlit.app`.

That's it — it's live. Update the demo link at the top of this README once
you have it.

**Alternative:** [Hugging Face Spaces](https://huggingface.co/spaces) also
supports Streamlit apps for free and works the same way (push repo, add
secret, done).

## Configuration

Everything tunable lives in `src/config.py`:

| Setting | Default | What it controls |
|---|---|---|
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE_WORDS` | `300` | Words per chunk |
| `CHUNK_OVERLAP_WORDS` | `60` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per question |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM used to generate answers |

## Running tests

```bash
pip install pytest
pytest tests/
```

## Roadmap / ideas for extending this

- **Persistent storage** — currently the index lives in Streamlit's session
  state and resets each session. Swap in a persisted vector DB (e.g.
  Chroma, Pinecone free tier) if you want notes to survive across visits.
- **Streaming responses** — Groq supports streaming; wire it up in
  `llm_client.py` for a typing effect instead of a full-block answer.
- **Follow-up context** — pass recent chat history into the prompt so
  "explain that more simply" works as a follow-up.
- **OCR support** — scanned PDFs currently return no text; add
  `pytesseract` + `pdf2image` as a fallback path in `pdf_processor.py`.

## Cost

- Embeddings + vector search: **$0**, runs locally.
- Groq API: **$0** on the free tier for normal study use — check current
  limits at [console.groq.com/settings/limits](https://console.groq.com/settings/limits).

## License

MIT — see [LICENSE](LICENSE).
