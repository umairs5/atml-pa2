from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.continue_train import run_grpo


def run_normalization_comparison(config_path: str, skip_training: bool = False):
    cfg = load_yaml(config_path)
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Task 3: GRPO Sequence Normalization Study ===")
    print(f"Comparing Canonical GRPO (1/T_k) vs Dr. GRPO (1/L_max) across {fork_updates} matched updates")

    forks = [
        {"name": "norm_canonical", "loss_type": "grpo", "desc": "Canonical GRPO (1/T_k)"},
        {"name": "norm_dr_grpo", "loss_type": "dr_grpo", "desc": "Dr. GRPO (1/L_max)"},
    ]

    fork_summaries = {}

    for fork in forks:
        run_name = fork["name"]
        loss_type = fork["loss_type"]
        out_dir = f"outputs/task3_grpo/{run_name}"

        if not skip_training:
            print(f"\n--- Running Fork: {fork['desc']} ---")
            summary = run_grpo(
                config_path=config_path,
                output=out_dir,
                updates=fork_updates,
                loss_type=loss_type,
                run_name=run_name,
            )
        else:
            summary_path = results_dir / f"{run_name}_train_summary.json"
            if summary_path.exists():
                summary = load_yaml(str(summary_path))
            else:
                summary = {}

        history_file = repo_path("outputs/task3_grpo") / f"{run_name}_train_history.jsonl"
        length_stats = {}
        if history_file.exists():
            records = read_jsonl(history_file)
            lengths = [r["response_length"] for r in records]
            rewards = [r["reward_mean"] for r in records]
            kls = [r["kl_mean"] for r in records]
            grad_norms = [r["grad_norm"] for r in records]
            entropies = [r["entropy"] for r in records]

            # Length-conditioned gradient weight analysis:
            # Under canonical GRPO, token weight is 1/T_k. Under Dr. GRPO, token weight is 1/L_max.
            # Compare the relative token weight allocation:
            max_comp_len = float(cfg.get("max_completion_length", 512))
            mean_len = float(np.mean(lengths)) if lengths else 1.0

            if loss_type == "grpo":
                weight_per_token_mean = 1.0 / max(mean_len, 1.0)
                length_bias_factor = max_comp_len / max(mean_len, 1.0)
            else:
                weight_per_token_mean = 1.0 / max_comp_len
                length_bias_factor = 1.0

            length_stats = {
                "mean_reward": float(np.mean(rewards)) if rewards else None,
                "final_reward": rewards[-1] if rewards else None,
                "mean_kl": float(np.mean(kls)) if kls else None,
                "mean_entropy": float(np.mean(entropies)) if entropies else None,
                "mean_response_length": mean_len,
                "max_grad_norm": float(np.max(grad_norms)) if grad_norms else None,
                "token_gradient_weight_proxy": round(weight_per_token_mean, 6),
                "relative_short_sequence_gradient_bias": round(length_bias_factor, 2),
            }

        fork_summaries[run_name] = {
            "name": run_name,
            "loss_type": loss_type,
            "description": fork["desc"],
            "updates": fork_updates,
            **length_stats,
        }

    comparison_file = results_dir / "normalization_comparison_summary.json"
    save_json(comparison_file, fork_summaries)

    print("\n=== GRPO Sequence Normalization Comparison Summary ===")
    print(f"{'Condition':<25} | {'Reward':<10} | {'KL':<10} | {'Length':<10} | {'Short-Bias Factor':<18}")
    print("-" * 80)
    for k, s in fork_summaries.items():
        rew = f"{s.get('mean_reward', 0.0):.3f}" if s.get('mean_reward') is not None else "N/A"
        kl = f"{s.get('mean_kl', 0.0):.4f}" if s.get('mean_kl') is not None else "N/A"
        l = f"{s.get('mean_response_length', 0.0):.1f}" if s.get('mean_response_length') is not None else "N/A"
        bias = f"{s.get('relative_short_sequence_gradient_bias', 1.0):.2f}x"
        print(f"{s['description']:<25} | {rew:<10} | {kl:<10} | {l:<10} | {bias:<18}")

    print(f"\nSaved normalization comparison summary to: {comparison_file}")
    return fork_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()
    run_normalization_comparison(args.config, skip_training=args.skip_training)


if __name__ == "__main__":
    main()
