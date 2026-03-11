from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUTS_DIR = BASE_DIR / "outputs" / "runs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT", ""),
    "user": os.getenv("SNOWFLAKE_USER", ""),
    "password": os.getenv("SNOWFLAKE_PASSWORD", ""),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "PODCASTIQ_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "PODCASTIQ"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "SEMANTIC"),
}

SEMANTIC_EMBEDDINGS_TABLE = "PODCASTIQ.SEMANTIC.SEM_CHUNK_EMBEDDINGS"

# Snowflake Cortex models
CORTEX_EMBED_MODEL = os.getenv("CORTEX_EMBED_MODEL", "snowflake-arctic-embed-m")
CORTEX_LLM_MODEL = os.getenv("CORTEX_LLM_MODEL", "llama3.1-70b")

TOP_K_INITIAL = 8
TOP_K_REFINED = 8
FINAL_EVIDENCE_K = 6

EVIDENCE_CHECK_PROMPT_PATH = PROMPTS_DIR / "evidence_check_prompt.txt"
ANSWER_PROMPT_PATH = PROMPTS_DIR / "answer_prompt.txt"
