from typing import List
from core.models import ChunkResult, FinalAnswer
from core.utils import format_chunk_for_prompt
from core.config import ANSWER_PROMPT_PATH
from core.cortex_llm import CortexService


class AnswerGenerator:
    def __init__(self, cortex_service: CortexService):
        self.cortex_service = cortex_service
        self.system_prompt = ANSWER_PROMPT_PATH.read_text(encoding="utf-8")

    def generate(self, question: str, chunks: List[ChunkResult]) -> FinalAnswer:
        evidence_text = "\n\n---\n\n".join(format_chunk_for_prompt(c) for c in chunks)

        full_prompt = (
            f"{self.system_prompt}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            f"Instructions:\n"
            f"- Use only the evidence above\n"
            f"- If evidence is incomplete, say so\n"
            f"- Mention specific episodes/channels when useful\n"
            f"- Do not invent facts\n"
        )

        answer_text = self.cortex_service.complete(full_prompt).strip()

        return FinalAnswer(
            answer_text=answer_text,
            cited_chunks=[c.chunk_id for c in chunks]
        )
