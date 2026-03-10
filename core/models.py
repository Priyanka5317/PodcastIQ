from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ChunkResult:
    chunk_id: str
    video_id: Optional[str]
    channel_name: str
    genre: Optional[str]
    episode_title: str
    publish_date: Optional[str]
    transcript_quality: Optional[str]
    chunk_start_sec: Optional[float]
    chunk_end_sec: Optional[float]
    word_count: Optional[int]
    youtube_url: Optional[str]
    chunk_text: str
    score: Optional[float] = None


@dataclass
class RetrievalPass:
    query: str
    top_k: int
    chunks: List[ChunkResult] = field(default_factory=list)


@dataclass
class EvidenceCheck:
    sufficient: bool
    missing_information: str
    refined_query: str


@dataclass
class FinalAnswer:
    answer_text: str
    cited_chunks: List[str] = field(default_factory=list)


@dataclass
class RealmRun:
    question: str
    initial_retrieval: RetrievalPass
    evidence_check: EvidenceCheck
    refined_retrieval: Optional[RetrievalPass]
    final_answer: FinalAnswer

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
