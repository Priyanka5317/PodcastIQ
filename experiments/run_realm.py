import argparse
import json
from pathlib import Path

from core.snowflake_client import SnowflakeClient
from core.cortex_llm import CortexService
from core.retriever import Retriever
from core.realm_engine import RealmEngine
from core.utils import save_run


def slugify(text: str) -> str:
    return "_".join(text.lower().strip().split())[:50]


def run_single(question: str):
    sf_client = SnowflakeClient()
    try:
        cortex_service = CortexService(sf_client)
        retriever = Retriever(sf_client)
        engine = RealmEngine(cortex_service, retriever)

        run = engine.run(question)

        print("\n=== QUESTION ===")
        print(run.question)

        print("\n=== INITIAL RETRIEVAL ===")
        for c in run.initial_retrieval.chunks:
            label = f"{c.channel_name} | {c.episode_title}"
            if c.score is not None:
                label += f" | score={c.score:.4f}"
            print(f"- {label}")

        print("\n=== EVIDENCE CHECK ===")
        print(f"sufficient: {run.evidence_check.sufficient}")
        print(f"missing_information: {run.evidence_check.missing_information}")
        print(f"refined_query: {run.evidence_check.refined_query}")
        print("\nraw_response:")
        print(run.evidence_check.raw_response)

        if run.refined_retrieval:
            print("\n=== REFINED RETRIEVAL ===")
            for c in run.refined_retrieval.chunks:
                label = f"{c.channel_name} | {c.episode_title}"
                if c.score is not None:
                    label += f" | score={c.score:.4f}"
                print(f"- {label}")

        print("\n=== FINAL ANSWER ===")
        print(run.final_answer.answer_text)

        save_run(run, slugify(question))
    finally:
        sf_client.close()


def run_batch(json_file: str):
    data = json.loads(Path(json_file).read_text(encoding="utf-8"))
    for item in data:
        question = item["question"]
        print(f"\n\n##### RUNNING: {question} #####")
        run_single(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Single question to run")
    parser.add_argument("--batch", type=str, help="Path to eval_questions.json")
    args = parser.parse_args()

    if args.question:
        run_single(args.question)
    elif args.batch:
        run_batch(args.batch)
    else:
        print("Provide --question or --batch")
