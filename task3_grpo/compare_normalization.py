from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import run_evaluation


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
            eval_summary = run_evaluation(
                config_path=config_path,
                adapter=out_dir,
                name=run_name,
            )
        else:
            summary_path = results_dir / f"{run_name}_train_summary.json"
            eval_path = results_dir / f"{run_name}_eval_summary.json"
            summary = load_json(summary_path) if summary_path.exists() else {}
            eval_summary = load_json(eval_path) if eval_path.exists() else {}

        history_file = repo_path("outputs/task3_grpo") / f"{run_name}_train_history.jsonl"
        length_stats = {}
        if history_file.exists():
            records = read_jsonl(history_file)
            lengths = [r["response_length"] for r in records]
            rewards = [r["reward_mean"] for r in records]
            kls = [r["kl_mean"] for r in records]
            grad_norms = [r["grad_norm"] for r in records]
            entropies = [r["entropy"] for r in records]

            mean_len = float(np.mean(lengths)) if lengths else None

            def average_optional(key):
                values = [float(r[key]) for r in records if r.get(key) is not None]
                return float(np.mean(values)) if values else None

            length_stats = {
                "mean_reward": float(np.mean(rewards)) if rewards else None,
                "final_reward": rewards[-1] if rewards else None,
                "mean_kl": float(np.mean(kls)) if kls else None,
                "mean_entropy": float(np.mean(entropies)) if entropies else None,
                "mean_response_length": mean_len,
                "max_grad_norm": float(np.max(grad_norms)) if grad_norms else None,
                "mean_short_token_weight": average_optional("short_token_weight"),
                "mean_long_token_weight": average_optional("long_token_weight"),
                "mean_short_to_long_token_weight": average_optional("short_to_long_token_weight"),
            }

        fork_summaries[run_name] = {
            "name": run_name,
            "loss_type": loss_type,
            "description": fork["desc"],
            "updates": fork_updates,
            "heldout_reward_mean": eval_summary.get("reward_mean"),
            "heldout_kl_mean": eval_summary.get("kl_mean"),
            "heldout_response_length_mean": eval_summary.get("response_length_mean"),
            **length_stats,
        }

    comparison_file = results_dir / "normalization_comparison_summary.json"
    save_json(comparison_file, fork_summaries)

    print("\n=== GRPO Sequence Normalization Comparison Summary ===")
    print(f"{'Condition':<25} | {'Held-out reward':<15} | {'Held-out KL':<12} | {'Held-out length':<15} | {'Short/long weight':<17}")
    print("-" * 105)
    for k, s in fork_summaries.items():
        rew = f"{s.get('heldout_reward_mean', 0.0):.3f}" if s.get('heldout_reward_mean') is not None else "N/A"
        kl = f"{s.get('heldout_kl_mean', 0.0):.4f}" if s.get('heldout_kl_mean') is not None else "N/A"
        length = f"{s.get('heldout_response_length_mean', 0.0):.1f}" if s.get('heldout_response_length_mean') is not None else "N/A"
        weight_ratio = s.get("mean_short_to_long_token_weight")
        ratio = f"{weight_ratio:.2f}x" if weight_ratio is not None else "N/A"
        print(f"{s['description']:<25} | {rew:<15} | {kl:<12} | {length:<15} | {ratio:<17}")

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
