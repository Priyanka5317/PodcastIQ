from typing import List
from core.models import ChunkResult
from core.snowflake_client import SnowflakeClient
from core.config import SEMANTIC_EMBEDDINGS_TABLE


class Retriever:
    def __init__(self, sf_client: SnowflakeClient):
        self.sf_client = sf_client

    @staticmethod
    def _vector_to_sql(vector: List[float]) -> str:
        joined = ",".join(str(float(x)) for x in vector)
        return f"[{joined}]"

    def retrieve(self, query_embedding: List[float], top_k: int = 8) -> List[ChunkResult]:
        vector_sql = self._vector_to_sql(query_embedding)

        sql = f"""
        SELECT
            CHUNK_ID,
            VIDEO_ID,
            CHANNEL_NAME,
            GENRE,
            EPISODE_TITLE,
            PUBLISH_DATE,
            TRANSCRIPT_QUALITY,
            CHUNK_START_SEC,
            CHUNK_END_SEC,
            WORD_COUNT,
            YOUTUBE_URL,
            CHUNK_TEXT,
            VECTOR_COSINE_SIMILARITY(EMBEDDING, {vector_sql}) AS SCORE
        FROM {SEMANTIC_EMBEDDINGS_TABLE}
        ORDER BY SCORE DESC
        LIMIT {top_k}
        """

        rows = self.sf_client.execute(sql)

        results = []
        for row in rows:
            results.append(
                ChunkResult(
                    chunk_id=row.get("CHUNK_ID"),
                    video_id=row.get("VIDEO_ID"),
                    channel_name=row.get("CHANNEL_NAME"),
                    genre=row.get("GENRE"),
                    episode_title=row.get("EPISODE_TITLE"),
                    publish_date=str(row.get("PUBLISH_DATE")) if row.get("PUBLISH_DATE") else None,
                    transcript_quality=row.get("TRANSCRIPT_QUALITY"),
                    chunk_start_sec=float(row["CHUNK_START_SEC"]) if row.get("CHUNK_START_SEC") is not None else None,
                    chunk_end_sec=float(row["CHUNK_END_SEC"]) if row.get("CHUNK_END_SEC") is not None else None,
                    word_count=int(row["WORD_COUNT"]) if row.get("WORD_COUNT") is not None else None,
                    youtube_url=row.get("YOUTUBE_URL"),
                    chunk_text=row.get("CHUNK_TEXT", ""),
                    score=float(row["SCORE"]) if row.get("SCORE") is not None else None,
                )
            )

        return results
