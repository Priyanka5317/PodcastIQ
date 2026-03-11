import json
from datetime import datetime
from typing import List
from core.models import ChunkResult, RealmRun
from core.config import OUTPUTS_DIR


def sec_to_hms(seconds: float) -> str:
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_chunk_for_prompt(chunk: ChunkResult) -> str:
    start = sec_to_hms(chunk.chunk_start_sec) if chunk.chunk_start_sec is not None else "N/A"
    end = sec_to_hms(chunk.chunk_end_sec) if chunk.chunk_end_sec is not None else "N/A"

    return (
        f"CHUNK_ID: {chunk.chunk_id}\n"
        f"CHANNEL: {chunk.channel_name}\n"
        f"EPISODE: {chunk.episode_title}\n"
        f"PUBLISH_DATE: {chunk.publish_date}\n"
        f"TIME_RANGE: {start} - {end}\n"
        f"URL: {chunk.youtube_url}\n"
        f"SCORE: {chunk.score}\n"
        f"TEXT:\n{chunk.chunk_text}\n"
    )


def save_run(run: RealmRun, slug: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUTS_DIR / f"{timestamp}_{slug}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"Saved run to: {json_path}")
