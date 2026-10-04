from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt: dict[str, list[dict]], k: int) -> list[dict]:
    """Return K-sized groups while keeping total cached completions fixed.

    Partitions each prompt's 8 completions into 8/k disjoint groups of size k.
    This guarantees that every group compares completions sampled for the same prompt,
    while keeping the total number of evaluated completions constant across K in {2, 4, 8}.
    """
    groups = []
    for pid, completions in by_prompt.items():
        # completions has length >= 8; use the first 8 completions
        first_8 = completions[:8]
        for start_idx in range(0, 8, k):
            subgroup = first_8[start_idx : start_idx + k]
            groups.append({
                "prompt_id": pid,
                "completions": subgroup,
                "rewards": [float(r["reward"]) for r in subgroup],
            })
    return groups


def run_group_size_study(config_path: str):
    cfg = load_yaml(config_path)
    group_cache_path = cfg.get("group_cache", "cached/grpo_k_cache.jsonl")
    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    by_prompt = load_k8_cache(group_cache_path)
    print(f"Loaded {len(by_prompt)} cached prompts with K=8 completions each from {group_cache_path}")

    # 1. Define prompt difficulty binning rule:
    # Based on the prompt's mean reward across all 8 cached completions.
    # Split at the median: 'easy' (>= median reward) vs 'hard' (< median reward).
    prompt_mean_rewards = {
        pid: float(np.mean([r["reward"] for r in group[:8]]))
        for pid, group in by_prompt.items()
    }
    median_reward = float(np.median(list(prompt_mean_rewards.values())))
    prompt_difficulty = {
        pid: ("easy" if m >= median_reward else "hard")
        for pid, m in prompt_mean_rewards.items()
    }

    print(f"Prompt Difficulty Binning (Median reward = {median_reward:.4f}):")
    print(f"  • Easy prompts (reward >= median): {sum(1 for v in prompt_difficulty.values() if v == 'easy')}")
    print(f"  • Hard prompts (reward < median) : {sum(1 for v in prompt_difficulty.values() if v == 'hard')}")

    results = {
        "metadata": {
            "total_prompts": len(by_prompt),
            "total_cached_generations": len(by_prompt) * 8,
            "median_reward_split": median_reward,
            "binning_rule": "Prompts split at median 8-completion mean reward into Easy vs Hard",
        },
        "group_sizes": {},
    }

    tol = 1e-6
    eps = 1e-6

    print("\n" + "=" * 90)
    print(f"{'K':<4} | {'Condition':<10} | {'Groups':<7} | {'Informative %':<15} | {'Reward Std':<12} | {'Signal Variance':<16}")
    print("-" * 90)

    for k in group_sizes:
        groups = regroup_equal_generation_budget(by_prompt, k)
        total_groups = len(groups)
        total_gens = total_groups * k

        # Overall metrics
        stds = [float(np.std(g["rewards"], ddof=0)) for g in groups]
        informative_flags = [s > tol for s in stds]
        inf_rate = float(np.mean(informative_flags))
        uninf_rate = 1.0 - inf_rate
        mean_std = float(np.mean(stds))

        # Relative advantage signal variance
        all_advantages = []
        for g, s in zip(groups, stds):
            r = np.array(g["rewards"], dtype=float)
            mu = np.mean(r)
            adv = (r - mu) / (s + eps)
            all_advantages.extend(adv.tolist())
        signal_var = float(np.var(all_advantages))

        k_summary = {
            "k": k,
            "total_groups": total_groups,
            "total_generations": total_gens,
            "informative_group_rate": inf_rate,
            "uninformative_group_rate": uninf_rate,
            "mean_reward_std": mean_std,
            "signal_variance": signal_var,
            "difficulty_bins": {},
        }

        print(f"{k:<4} | {'Overall':<10} | {total_groups:<7} | {inf_rate*100:<14.2f}% | {mean_std:<12.4f} | {signal_var:<16.4f}")

        # Metrics broken down by difficulty bin
        for diff in ["easy", "hard"]:
            diff_groups = [g for g in groups if prompt_difficulty[g["prompt_id"]] == diff]
            diff_stds = [float(np.std(g["rewards"], ddof=0)) for g in diff_groups]
            diff_inf = [s > tol for s in diff_stds]
            diff_inf_rate = float(np.mean(diff_inf))
            diff_mean_std = float(np.mean(diff_stds))

            diff_advs = []
            for g, s in zip(diff_groups, diff_stds):
                r = np.array(g["rewards"], dtype=float)
                mu = np.mean(r)
                adv = (r - mu) / (s + eps)
                diff_advs.extend(adv.tolist())
            diff_signal_var = float(np.var(diff_advs))

            k_summary["difficulty_bins"][diff] = {
                "num_groups": len(diff_groups),
                "informative_group_rate": diff_inf_rate,
                "uninformative_group_rate": 1.0 - diff_inf_rate,
                "mean_reward_std": diff_mean_std,
                "signal_variance": diff_signal_var,
            }

            print(f"     | {diff.capitalize():<10} | {len(diff_groups):<7} | {diff_inf_rate*100:<14.2f}% | {diff_mean_std:<12.4f} | {diff_signal_var:<16.4f}")

        results["group_sizes"][f"k_{k}"] = k_summary
        print("-" * 90)

    out_file = results_dir / "group_size_analysis.json"
    save_json(out_file, results)
    print(f"\nSaved group-size analysis to: {out_file}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_group_size_study(args.config)


if __name__ == "__main__":
    main()
