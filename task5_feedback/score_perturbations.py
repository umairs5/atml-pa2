from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from common.models import clear_gpu
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward


EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
COMPARISON_VARIANTS = (
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
)


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def rate_summary(labels: list[str]) -> dict:
    counts = Counter(labels)
    n = len(labels)
    return {
        "pairs": n,
        "better_response_rate": float(counts["A"] / n) if n else 0.0,
        "tie_rate": float(counts["TIE"] / n) if n else 0.0,
        "wrong_preference_rate": float(counts["B"] / n) if n else 0.0,
        "better_response_count": int(counts["A"]),
        "tie_count": int(counts["TIE"]),
        "wrong_preference_count": int(counts["B"]),
    }


def task5_results_dir(cfg: dict):
    return repo_path(cfg.get("results_dir", "results")) / "task5_feedback"


def score_diagnostics(cfg: dict) -> dict:
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    outdir = task5_results_dir(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    judge = PairwiseAIJudge(cfg, outdir / "diagnostic_judge_cache.json")
    records = []

    for problem_id, variants in sorted(groups.items(), key=lambda item: item[0]):
        clean = variants["clean_correct"]
        gold = str(clean["gold_final"])
        clean_exact = exact_reward(clean["response"], gold)
        for variant_name in COMPARISON_VARIANTS:
            variant = variants[variant_name]
            variant_exact = exact_reward(variant["response"], gold)
            verifier_preference = "A" if clean_exact > variant_exact else "B" if variant_exact > clean_exact else "TIE"
            records.append({
                "problem_id": problem_id,
                "comparison": f"clean_correct_vs_{variant_name}",
                "variant_type": variant_name,
                "verifier_preference": verifier_preference,
                "ai_preference": judge.compare(str(clean["question"]), clean["response"], variant["response"]),
                "clean_exact_reward": clean_exact,
                "variant_exact_reward": variant_exact,
            })

    clear_gpu(judge.model)
    by_variant = {}
    for variant_name in COMPARISON_VARIANTS:
        subset = [row for row in records if row["variant_type"] == variant_name]
        by_variant[variant_name] = {
            "rlvr": rate_summary([row["verifier_preference"] for row in subset]),
            "rlaif": rate_summary([row["ai_preference"] for row in subset]),
        }

    outcome_rows = [row for row in records if row["variant_type"] in {"good_reasoning_wrong_final", "gold_distractor_wrong_final"}]
    raw_rows = [row for variants in groups.values() for row in variants.values()]
    variant_rewards = {}
    for variant_name in sorted(EXPECTED_VARIANTS):
        rows = [row for row in raw_rows if row["variant_type"] == variant_name]
        variant_rewards[variant_name] = float(sum(exact_reward(row["response"], str(row["gold_final"])) for row in rows) / max(len(rows), 1))

    summary = {
        "num_problems": len(groups),
        "pairs_per_comparison": len(groups),
        "by_variant": by_variant,
        "reasoning_sensitivity": by_variant["corrupt_reasoning_correct_final"],
        "outcome_sensitivity": {
            "rlvr": rate_summary([row["verifier_preference"] for row in outcome_rows]),
            "rlaif": rate_summary([row["ai_preference"] for row in outcome_rows]),
        },
        "filler_susceptibility": by_variant["persuasive_filler_correct"],
        "distractor_robustness": by_variant["gold_distractor_wrong_final"],
        "exact_reward_by_variant": variant_rewards,
    }
    write_jsonl(outdir / "diagnostic_pair_scores.jsonl", records)
    save_json(outdir / "diagnostic_summary.json", summary)
    print(f"Saved diagnostic scores to: {outdir / 'diagnostic_pair_scores.jsonl'}")
    print(f"Saved diagnostic summary to: {outdir / 'diagnostic_summary.json'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    summary = score_diagnostics(load_yaml(args.config))
    print(f"Scored {summary['num_problems']} diagnostic problems")


if __name__ == "__main__":
    main()
