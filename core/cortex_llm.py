import json
from typing import List
from core.config import CORTEX_EMBED_MODEL, CORTEX_LLM_MODEL
from core.snowflake_client import SnowflakeClient


class CortexService:
    def __init__(self, sf_client: SnowflakeClient):
        self.sf_client = sf_client
        self.embed_model = CORTEX_EMBED_MODEL
        self.llm_model = CORTEX_LLM_MODEL

    def embed_text(self, text: str) -> List[float]:
        sql = f"""
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(%s, %s) AS EMBEDDING
        """
        result = self.sf_client.execute_scalar(sql, (self.embed_model, text))

        # Snowflake may return Python list, JSON string, or variant-like text
        if isinstance(result, list):
            return [float(x) for x in result]

        if isinstance(result, str):
            parsed = json.loads(result)
            return [float(x) for x in parsed]

        raise ValueError(f"Unexpected embedding response type: {type(result)}")

    def complete(self, prompt: str) -> str:
        sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS RESPONSE
        """
        result = self.sf_client.execute_scalar(sql, (self.llm_model, prompt))

        if result is None:
            return ""

        return str(result)
