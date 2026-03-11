from typing import Optional, List

from core.config import TOP_K_INITIAL, TOP_K_REFINED
from core.models import RetrievalPass, EvidenceCheck, RealmRun, ChunkResult
from core.retriever import Retriever
from core.evidence_selector import EvidenceSelector
from core.answer_generator import AnswerGenerator
from core.utils import format_chunk_for_prompt
from core.config import EVIDENCE_CHECK_PROMPT_PATH
from core.cortex_llm import CortexService


class RealmEngine:
    def __init__(self, cortex_service: CortexService, retriever: Retriever):
        self.cortex_service = cortex_service
        self.retriever = retriever
        self.selector = EvidenceSelector()
        self.answer_generator = AnswerGenerator(cortex_service)
        self.evidence_prompt = EVIDENCE_CHECK_PROMPT_PATH.read_text(encoding="utf-8")

    def _check_evidence(self, question: str, chunks: List[ChunkResult]) -> EvidenceCheck:
        evidence_text = "\n\n---\n\n".join(format_chunk_for_prompt(c) for c in chunks)

        full_prompt = (
            f"{self.evidence_prompt}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED EVIDENCE:\n{evidence_text}\n\n"
            f"Return exactly this format:\n"
            f"sufficient: yes/no\n"
            f"missing_information: <short text>\n"
            f"refined_query: <short retrieval query>\n"
        )

        text = self.cortex_service.complete(full_prompt).strip()

        sufficient = "sufficient: yes" in text.lower()
        missing_information = ""
        refined_query = question

        for line in text.splitlines():
            lower_line = line.lower()
            if lower_line.startswith("missing_information:"):
                missing_information = line.split(":", 1)[1].strip()
            elif lower_line.startswith("refined_query:"):
                refined_query = line.split(":", 1)[1].strip()

        return EvidenceCheck(
            sufficient=sufficient,
            missing_information=missing_information,
            refined_query=refined_query,
            raw_response=text
        )

    def run(self, question: str) -> RealmRun:
        # Initial retrieval
        initial_embedding = self.cortex_service.embed_text(question)
        initial_chunks = self.retriever.retrieve(initial_embedding, top_k=TOP_K_INITIAL)
        initial_pass = RetrievalPass(query=question, top_k=TOP_K_INITIAL, chunks=initial_chunks)

        # Evidence check
        evidence_check = self._check_evidence(question, initial_chunks)

        refined_pass: Optional[RetrievalPass] = None
        combined_chunks = list(initial_chunks)

        # Refined retrieval if needed
        if not evidence_check.sufficient and evidence_check.refined_query.strip():
            refined_embedding = self.cortex_service.embed_text(evidence_check.refined_query)
            refined_chunks = self.retriever.retrieve(refined_embedding, top_k=TOP_K_REFINED)
            refined_pass = RetrievalPass(
                query=evidence_check.refined_query,
                top_k=TOP_K_REFINED,
                chunks=refined_chunks
            )
            combined_chunks.extend(refined_chunks)

        selected_chunks = self.selector.select(combined_chunks)
        final_answer = self.answer_generator.generate(question, selected_chunks)

        return RealmRun(
            question=question,
            initial_retrieval=initial_pass,
            evidence_check=evidence_check,
            refined_retrieval=refined_pass,
            final_answer=final_answer,
        )
