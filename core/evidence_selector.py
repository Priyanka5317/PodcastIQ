from typing import List
from core.models import ChunkResult
from core.config import FINAL_EVIDENCE_K


class EvidenceSelector:
    def select(self, chunks: List[ChunkResult], k: int = FINAL_EVIDENCE_K) -> List[ChunkResult]:
        seen = set()
        unique_chunks = []

        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.score if x.score is not None else 0.0,
            reverse=True
        )

        for chunk in sorted_chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_chunks.append(chunk)

        return unique_chunks[:k]
