"""
llm_client.py
Thin, defensive wrapper around the Groq chat completions API.
Isolated here so swapping providers later (Gemini, OpenAI, a local model
via Ollama) only means editing this one file.
"""

import requests

from src.config import GROQ_API_URL, GROQ_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS, SYSTEM_PROMPT
from src.pdf_processor import Chunk


class LLMError(Exception):
    """Raised when the LLM call fails for any reason. Message is user-safe."""


def _format_context(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c.source}, Page {c.page}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, context_chunks: list[Chunk], api_key: str) -> str:
    """
    Sends the question + retrieved context to Groq and returns the answer text.
    Raises LLMError with a user-friendly message on any failure — callers
    should catch this and display it rather than letting it crash the app.
    """
    if not api_key:
        raise LLMError(
            "No Groq API key configured. Add GROQ_API_KEY to your Streamlit "
            "secrets or environment variables."
        )

    if not context_chunks:
        raise LLMError(
            "No relevant material found in your uploaded notes for this question."
        )

    context = _format_context(context_chunks)
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Lecture notes excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        "temperature": LLM_TEMPERATURE,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            GROQ_API_URL, headers=headers, json=payload, timeout=LLM_TIMEOUT_SECONDS
        )
    except requests.exceptions.Timeout:
        raise LLMError("The request to the AI model timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise LLMError("Couldn't connect to the AI service. Check your internet connection.")

    if response.status_code == 401:
        raise LLMError("Groq API key was rejected. Double-check it's correct and active.")
    if response.status_code == 429:
        raise LLMError("Rate limit hit on the free tier. Wait a moment and try again.")
    if response.status_code != 200:
        raise LLMError(f"AI service returned an error ({response.status_code}). Please try again.")

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except (KeyError, ValueError, IndexError):
        raise LLMError("Received an unexpected response from the AI service.")
