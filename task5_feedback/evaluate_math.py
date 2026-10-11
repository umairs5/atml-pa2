from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


POLICIES = ("sft", "rlvr", "rlaif")


def policy_specs(cfg):
    return {"sft": None, "rlvr": cfg["policies"]["rlvr"], "rlaif": cfg["policies"]["rlaif"]}


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(f"Unsupported dataset: {dataset}")


def task5_results_dir(cfg: dict) -> Path:
    return repo_path(cfg.get("results_dir", "results")) / "task5_feedback"


def row_id(row: dict) -> str:
    return str(row.get("prompt_id", row.get("source_index")))


def preference_from_exact(a_reward: float, b_reward: float) -> str:
    if a_reward > b_reward:
        return "A"
    if b_reward > a_reward:
        return "B"
    return "TIE"


def summarize_policy(records: list[dict]) -> dict:
    exact = np.asarray([row["exact_reward"] for row in records], dtype=float)
    compliant = np.asarray([row["format_compliant"] for row in records], dtype=float)
    lengths = np.asarray([row["response_length"] for row in records], dtype=float)
    return {
        "num_examples": int(len(records)),
        "exact_accuracy": float(exact.mean()) if len(exact) else 0.0,
        "format_compliance": float(compliant.mean()) if len(compliant) else 0.0,
        "mean_response_length": float(lengths.mean()) if len(lengths) else 0.0,
        "response_length_std": float(lengths.std()) if len(lengths) else 0.0,
    }


def summarize_pairwise(records: list[dict]) -> dict:
    labels = [row["ai_preference"] for row in records]
    counts = Counter(labels)
    agreement = np.mean([row["ai_preference"] == row["verifier_preference"] for row in records]) if records else 0.0
    return {
        "comparisons": int(len(records)),
        "ai_wins": int(counts["A"]),
        "ai_losses": int(counts["B"]),
        "ai_ties": int(counts["TIE"]),
        "ai_win_rate_vs_sft": float((counts["A"] + 0.5 * counts["TIE"]) / len(records)) if records else 0.0,
        "verifier_judge_agreement": float(agreement),
    }


def generate_policy_records(cfg: dict, rows: list[dict], tokenizer, policy_name: str, batch_size: int) -> list[dict]:
    """Generate one reproducible response for every fixed evaluation prompt."""
    set_seed(int(cfg["seed"]))
    model = load_policy(cfg, adapter_path=policy_specs(cfg)[policy_name], trainable=False)
    records = []
    generation = cfg.get("generation", {})
    print(f"Generating {len(rows)} {policy_name} responses...")
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        messages = [prompt_messages(row) for row in chunk]
        with torch.no_grad():
            generated = batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=messages,
                max_prompt_length=int(cfg.get("max_prompt_length", 256)),
                max_new_tokens=int(cfg.get("math_max_new_tokens", 512)),
                temperature=float(generation.get("temperature", 0.7)),
                top_p=float(generation.get("top_p", 0.9)),
                do_sample=bool(generation.get("do_sample", True)),
            )
        for offset, row in enumerate(chunk):
            response = generated["responses"][offset]
            gold = str(row["gold_final"])
            records.append({
                "dataset": row.get("dataset"),
                "source_index": row.get("source_index"),
                "prompt_id": row_id(row),
                "question": row.get("question", ""),
                "gold_final": gold,
                "policy": policy_name,
                "response": response,
                "predicted_final": extract_designated_final(response),
                "exact_reward": exact_reward(response, gold),
                "format_compliant": float(extract_designated_final(response) is not None),
                "response_length": int(generated["response_lengths"][offset]),
                "terminated_with_eos": bool(generated["terminated_with_eos"][offset]),
                "truncated": bool(generated["truncated"][offset]),
            })
        if start + batch_size >= len(rows) or (start // batch_size + 1) % 10 == 0:
            print(f"  {min(start + batch_size, len(rows))}/{len(rows)}")
    clear_gpu(model)
    return records


def load_or_generate(cfg: dict, dataset: str, rows: list[dict], tokenizer, batch_size: int, overwrite: bool) -> list[dict]:
    path = task5_results_dir(cfg) / f"{dataset}_generations.jsonl"
    expected = len(rows)
    if path.exists() and not overwrite:
        cached = read_jsonl(path)
        by_policy = Counter(row.get("policy") for row in cached)
        if set(by_policy) == set(POLICIES) and all(by_policy[name] == expected for name in POLICIES):
            print(f"Using cached generations: {path}")
            return cached
        print("Cached generations are incomplete; regenerating them.")
    records = []
    for policy_name in POLICIES:
        records.extend(generate_policy_records(cfg, rows, tokenizer, policy_name, batch_size))
    write_jsonl(path, records)
    print(f"Saved generations to: {path}")
    return records


def make_pairwise_records(cfg: dict, dataset: str, rows: list[dict], generations: list[dict]) -> list[dict]:
    by_policy_id = {(row["policy"], row["prompt_id"]): row for row in generations}
    judge = PairwiseAIJudge(cfg, task5_results_dir(cfg) / f"{dataset}_judge_cache.json")
    records = []
    for policy_name in ("rlvr", "rlaif"):
        for row in rows:
            prompt_id = row_id(row)
            candidate = by_policy_id[(policy_name, prompt_id)]
            baseline = by_policy_id[("sft", prompt_id)]
            ai_preference = judge.compare(str(row.get("question", "")), candidate["response"], baseline["response"])
            records.append({
                "dataset": dataset,
                "prompt_id": prompt_id,
                "policy": policy_name,
                "baseline_policy": "sft",
                "ai_preference": ai_preference,
                "verifier_preference": preference_from_exact(candidate["exact_reward"], baseline["exact_reward"]),
                "candidate_exact_reward": candidate["exact_reward"],
                "baseline_exact_reward": baseline["exact_reward"],
            })
    clear_gpu(judge.model)
    return records


def run_evaluation(config_path: str, dataset: str, batch_size: int = 2, overwrite: bool = False) -> dict:
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    outdir = task5_results_dir(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    generations = load_or_generate(cfg, dataset, rows, tokenizer, batch_size, overwrite)
    pairwise = make_pairwise_records(cfg, dataset, rows, generations)
    policy_summaries = {
        policy_name: summarize_policy([row for row in generations if row["policy"] == policy_name])
        for policy_name in POLICIES
    }
    pairwise_summaries = {
        policy_name: summarize_pairwise([row for row in pairwise if row["policy"] == policy_name])
        for policy_name in ("rlvr", "rlaif")
    }
    summary = {"dataset": dataset, "policies": policy_summaries, "pairwise_vs_sft": pairwise_summaries}
    write_jsonl(outdir / f"{dataset}_pairwise.jsonl", pairwise)
    save_json(outdir / f"{dataset}_summary.json", summary)
    print(f"Saved Task 5 {dataset} summary to: {outdir / f'{dataset}_summary.json'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    summary = run_evaluation(args.config, args.dataset, args.batch_size, args.overwrite)
    for policy_name, values in summary["policies"].items():
        print(f"{policy_name}: exact={values['exact_accuracy']:.3f}, format={values['format_compliance']:.3f}, length={values['mean_response_length']:.1f}")
    for policy_name, values in summary["pairwise_vs_sft"].items():
        print(f"{policy_name} vs SFT: AI win={values['ai_win_rate_vs_sft']:.3f}, verifier/judge agreement={values['verifier_judge_agreement']:.3f}")


if __name__ == "__main__":
    main()
