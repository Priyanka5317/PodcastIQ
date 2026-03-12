import json
import os
import re

TRANSCRIPT_PATH = os.path.join("data", "sample_transcript.txt")
CHUNKS_PATH = os.path.join("data", "processed", "chunks.json")
OUTPUT_PATH = os.path.join("outputs", "retrieval_output.json")

GENERIC_QUERY_WORDS = {
    "company", "companies", "organization", "organizations",
    "brand", "brands", "product", "products",
    "podcast", "transcript", "mentioned", "names",
    "what", "which", "are", "the", "in", "of", "about"
}

KNOWN_ENTITIES = [
    "openai", "anthropic", "cursor", "claude", "lovable", "grok", "xai"
]

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    chunk_id = 1

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_id": f"chunk_{chunk_id}",
                "text": chunk
            })
            chunk_id += 1
        start += chunk_size - overlap

    return chunks

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

def score_chunk(query, chunk_text_value):
    query_tokens = [
        token for token in tokenize(query)
        if token not in GENERIC_QUERY_WORDS
    ]

    chunk_tokens = tokenize(chunk_text_value)
    chunk_token_set = set(chunk_tokens)

    lexical_score = sum(1 for token in query_tokens if token in chunk_token_set)

    chunk_lower = chunk_text_value.lower()

    entity_hits = sum(1 for ent in KNOWN_ENTITIES if ent in chunk_lower)

    generic_hits = sum(
        1 for word in ["company", "companies", "brand", "brands", "product", "products"]
        if word in chunk_lower
    )

    score = lexical_score
    score += entity_hits * 3

    if generic_hits > 0 and entity_hits == 0:
        score -= 2

    return score

def retrieve_top_chunks(query, chunks, top_k=3):
    scored = []
    for chunk in chunks:
        score = score_chunk(query, chunk["text"])
        scored.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def main():
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript = f.read()

    chunks = chunk_text(transcript)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    query = input("Enter your question: ")

    top_chunks = retrieve_top_chunks(query, chunks, top_k=3)

    print("\nTop retrieved chunks:\n")
    for item in top_chunks:
        print(f"{item['chunk_id']} | score={item['score']}")
        print(item["text"][:300])
        print("-" * 60)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(top_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved chunks to: {os.path.abspath(CHUNKS_PATH)}")
    print(f"Saved retrieval output to: {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main()