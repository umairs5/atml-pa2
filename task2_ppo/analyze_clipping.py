from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task2_ppo.continue_train import run_ppo


def load_cached_rollouts(path):
    p = repo_path(path)
    if not p.exists():
        print(f"Notice: Cache file '{path}' not found on this machine.")
        return []

    rows = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def analyze_cached_batch(rows: list[dict], clip_values: list[float], results_dir: Path):
    if not rows:
        return {}

    print(f"\n--- Analyzing {len(rows)} cached rollouts for clipping fractions ---")
    analysis = {}

    all_old_logp = []
    all_ref_logp = []
    all_adv = []

    for r in rows:
        old_lp = torch.as_tensor(r["old_logprobs"]).float()
        ref_lp = torch.as_tensor(r["ref_logprobs"]).float()
        all_old_logp.append(old_lp)
        all_ref_logp.append(ref_lp)
        if "advantages" in r:
            all_adv.append(torch.as_tensor(r["advantages"]).float())

    cat_old = torch.cat(all_old_logp)
    cat_ref = torch.cat(all_ref_logp)

    # Use the drift between old and reference policy (or candidate policy) to measure clipping sensitivity
    drift_ratio = torch.exp(cat_old - cat_ref)

    for eps in clip_values:
        affected = ((drift_ratio < (1.0 - eps)) | (drift_ratio > (1.0 + eps))).float()
        clip_frac = float(affected.mean().item())

        stat = {
            "clip_epsilon": eps,
            "clip_fraction": clip_frac,
            "mean_ratio": float(drift_ratio.mean().item()),
            "std_ratio": float(drift_ratio.std().item()),
            "total_tokens_evaluated": len(cat_old),
        }

        if all_adv:
            cat_adv = torch.cat(all_adv)
            surr1 = drift_ratio * cat_adv
            surr2 = drift_ratio.clamp(1.0 - eps, 1.0 + eps) * cat_adv
            obj = torch.minimum(surr1, surr2).mean().item()
            stat["surrogate_objective"] = float(obj)

        analysis[f"eps_{eps}"] = stat
        print(f"  eps={eps:<4.2f} | Affected Token Clip Fraction: {clip_frac * 100:.2f}%")

    out_file = results_dir / "cached_clipping_analysis.json"
    save_json(out_file, analysis)
    print(f"Saved cached clipping analysis to: {out_file}")
    return analysis


def run_clipping_forks(config_path: str, skip_forks: bool = False):
    cfg = load_yaml(config_path)
    clip_values = [float(x) for x in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Analyze cached rollouts if available
    cache_path = cfg.get("cached_rollouts", "cached/ppo_rollout.pt")
    cached_rows = load_cached_rollouts(cache_path)
    if cached_rows:
        analyze_cached_batch(cached_rows, clip_values, results_dir)

    if skip_forks:
        return

    # 2. Run short forks for each epsilon
    print(f"\n=== Running PPO Clipping Epsilon Ablation Forks ===")
    print(f"Epsilon values: {clip_values}")
    print(f"Updates per fork: {fork_updates}")

    fork_summaries = {}

    for eps in clip_values:
        run_name = f"clip_eps_{eps:.2f}".replace(".", "_")
        out_dir = f"outputs/task2_ppo/{run_name}"
        print(f"\n--- Starting fork: {run_name} (eps={eps}) ---")
        run_ppo(
            config_path=config_path,
            output=out_dir,
            updates=fork_updates,
            clip_epsilon=eps,
            run_name=run_name,
        )

        history_file = repo_path(out_dir).parent / f"{run_name}_train_history.jsonl"
        if history_file.exists():
            records = read_jsonl(history_file)
            rewards = [r["reward_mean"] for r in records]
            kls = [r["kl_mean"] for r in records]
            entropies = [r["entropy"] for r in records]
            clip_fracs = [r["clip_fraction"] for r in records]
            grad_norms = [r["grad_norm"] for r in records]

            fork_summaries[f"eps_{eps}"] = {
                "clip_epsilon": eps,
                "final_reward": rewards[-1] if rewards else None,
                "mean_reward": float(np.mean(rewards)) if rewards else None,
                "mean_kl": float(np.mean(kls)) if kls else None,
                "mean_entropy": float(np.mean(entropies)) if entropies else None,
                "mean_clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else None,
                "max_grad_norm": float(np.max(grad_norms)) if grad_norms else None,
                "history_file": str(history_file),
            }

    summary_file = results_dir / "clipping_ablation_summary.json"
    save_json(summary_file, fork_summaries)

    print("\n=== Clipping Epsilon Ablation Summary ===")
    print(f"{'Epsilon':<8} | {'Mean Reward':<12} | {'Mean KL':<10} | {'Mean Entropy':<12} | {'Mean Clip Frac':<16} | {'Max Grad Norm':<14}")
    print("-" * 85)
    for k, s in fork_summaries.items():
        print(f"{s['clip_epsilon']:<8.2f} | {s['mean_reward']:<12.3f} | {s['mean_kl']:<10.4f} | {s['mean_entropy']:<12.3f} | {s['mean_clip_fraction']:<16.4f} | {s['max_grad_norm']:<14.3f}")
    print(f"\nSaved clipping ablation summary to: {summary_file}")
    return fork_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-forks", action="store_true")
    args = ap.parse_args()
    run_clipping_forks(args.config, skip_forks=args.skip_forks)


if __name__ == "__main__":
    main()
