"""
config.py
Central place for every tunable constant in the app.
Change chunking, model choice, or retrieval depth here — nowhere else.
"""

# --- Embeddings (local, free, no API calls) ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast on CPU, good quality/speed tradeoff

# --- Chunking ---
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 60

# --- Retrieval ---
TOP_K = 5  # number of chunks fed to the LLM per question

# --- LLM (Groq free tier) ---
#GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "You are a patient, precise personal tutor. Answer the student's question "
    "using ONLY the provided excerpts from their own lecture notes below. "
    "Rules:\n"
    "1. If the excerpts don't contain the answer, say so plainly — never invent facts.\n"
    "2. Explain concepts the way a good tutor would: build intuition, don't just "
    "restate definitions.\n"
    "3. When you reference something, mention which source/page it came from.\n"
    "4. Keep answers focused and skimmable — use short paragraphs or bullet points "
    "for multi-part explanations."
)
