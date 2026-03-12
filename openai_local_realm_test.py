import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
TRANSCRIPT_PATH = Path("data/sample_transcript.txt")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 30
TOP_K = 5

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

EMBED_CACHE_PATH = OUTPUT_DIR / "embedded_chunks_cache.json"


# -----------------------------
# Helpers
# -----------------------------
def load_transcript(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Transcript file not found: {path}\n"
            f"Create it and paste transcript text into it."
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Transcript file is empty.")
    return text


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[Dict[str, Any]]:
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 1

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text_value = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"chunk_{chunk_id}",
            "start_word": start,
            "end_word": end,
            "text": chunk_text_value
        })

        if end == len(words):
            break

        start += max(1, chunk_size - overlap)
        chunk_id += 1

    return chunks


def get_single_embedding(client: OpenAI, text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_batch_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def magnitude(vec: List[float]) -> float:
    return sum(x * x for x in vec) ** 0.5


def cosine_similarity(a: List[float], b: List[float]) -> float:
    mag_a = magnitude(a)
    mag_b = magnitude(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot_product(a, b) / (mag_a * mag_b)


def save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cached_embeddings(path: Path) -> List[Dict[str, Any]] | None:
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def embed_chunks(client: OpenAI, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cached = load_cached_embeddings(EMBED_CACHE_PATH)

    if cached:
        print(f"\nLoaded cached embeddings from: {EMBED_CACHE_PATH}")
        return cached

    print(f"\nEmbedding {len(chunks)} chunks with OpenAI in batch...")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_batch_embeddings(client, texts)

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            **chunk,
            "embedding": embedding
        })

    save_json(embedded_chunks, EMBED_CACHE_PATH)
    print(f"Saved embedding cache to: {EMBED_CACHE_PATH}")

    return embedded_chunks


def retrieve_top_chunks(
    client: OpenAI,
    question: str,
    embedded_chunks: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    print("\nEmbedding the question...")
    question_embedding = get_single_embedding(client, question)

    scored = []
    for chunk in embedded_chunks:
        score = cosine_similarity(question_embedding, chunk["embedding"])
        scored.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score,
            "start_word": chunk["start_word"],
            "end_word": chunk["end_word"]
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def generate_grounded_answer(client: OpenAI, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        [f"{c['chunk_id']} (score={c['score']:.4f}):\n{c['text']}" for c in retrieved_chunks]
    )

    system_prompt = (
        "You are a grounded question-answering assistant.\n"
        "Answer ONLY using the provided retrieved transcript chunks.\n"
        "If the answer is not supported by the retrieved chunks, say:\n"
        "'I could not find enough evidence in the retrieved transcript chunks.'\n"
        "Do not invent facts.\n"
        "Keep the answer clear and concise."
    )

    user_prompt = f"""
Question:
{question}

Retrieved transcript chunks:
{context}

Instructions:
- Answer using only the evidence above.
- If useful, mention which chunk supports the answer.
- If the retrieved evidence is weak, say so clearly.
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found.\n"
            "Add it to your .env file like:\n"
            "OPENAI_API_KEY=your_key_here"
        )

    client = OpenAI(api_key=api_key)

    transcript = load_transcript(TRANSCRIPT_PATH)
    chunks = chunk_text(
        transcript,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=CHUNK_OVERLAP_WORDS
    )

    print(f"\nLoaded transcript from: {TRANSCRIPT_PATH}")
    print(f"Created {len(chunks)} chunks.")

    embedded_chunks = embed_chunks(client, chunks)

    embedded_preview = [
        {
            "chunk_id": c["chunk_id"],
            "start_word": c["start_word"],
            "end_word": c["end_word"],
            "text": c["text"]
        }
        for c in embedded_chunks
    ]
    save_json(embedded_preview, OUTPUT_DIR / "embedded_chunks_preview.json")

    question = input("\nEnter your question: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    top_chunks = retrieve_top_chunks(client, question, embedded_chunks, top_k=TOP_K)

    print("\nTop retrieved chunks:\n")
    for chunk in top_chunks:
        preview = chunk["text"][:350].replace("\n", " ")
        print(f"{chunk['chunk_id']} | similarity={chunk['score']:.4f}")
        print(preview)
        print("-" * 60)

    print("\nGenerating grounded answer with OpenAI...\n")
    answer = generate_grounded_answer(client, question, top_chunks)

    print("Final Answer:\n")
    print(answer)

    retrieval_output = {
        "question": question,
        "top_chunks": top_chunks,
        "final_answer": answer
    }

    save_json(retrieval_output, OUTPUT_DIR / "openai_retrieval_output.json")
    print(f"\nSaved output to: {OUTPUT_DIR / 'openai_retrieval_output.json'}")


if __name__ == "__main__":
    main()