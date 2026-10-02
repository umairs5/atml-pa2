from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task2_ppo.continue_train import run_ppo


def run_kl_ablation(config_path: str):
    cfg = load_yaml(config_path)
    kl_values = [float(x) for x in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running PPO KL Beta Ablation ===")
    print(f"KL beta values: {kl_values}")
    print(f"Updates per fork: {fork_updates}")

    fork_summaries = {}

    for beta in kl_values:
        run_name = f"kl_beta_{beta:.2f}".replace(".", "_")
        out_dir = f"outputs/task2_ppo/{run_name}"
        print(f"\n--- Starting fork: {run_name} (beta={beta}) ---")
        run_ppo(
            config_path=config_path,
            output=out_dir,
            updates=fork_updates,
            kl_beta=beta,
            run_name=run_name,
        )

        history_file = repo_path(out_dir).parent / f"{run_name}_train_history.jsonl"
        if history_file.exists():
            records = read_jsonl(history_file)
            rewards = [r["reward_mean"] for r in records]
            kls = [r["kl_mean"] for r in records]
            entropies = [r["entropy"] for r in records]
            lengths = [r["response_length"] for r in records]
            grad_norms = [r["grad_norm"] for r in records]

            fork_summaries[f"beta_{beta}"] = {
                "kl_beta": beta,
                "final_reward": rewards[-1] if rewards else None,
                "mean_reward": float(np.mean(rewards)) if rewards else None,
                "final_kl": kls[-1] if kls else None,
                "mean_kl": float(np.mean(kls)) if kls else None,
                "mean_entropy": float(np.mean(entropies)) if entropies else None,
                "mean_response_length": float(np.mean(lengths)) if lengths else None,
                "max_grad_norm": float(np.max(grad_norms)) if grad_norms else None,
                "history_file": str(history_file),
            }

    summary_file = results_dir / "kl_ablation_summary.json"
    save_json(summary_file, fork_summaries)

    print("\n=== KL Beta Ablation Summary ===")
    print(f"{'Beta':<8} | {'Mean Reward':<12} | {'Final KL':<10} | {'Mean Entropy':<12} | {'Mean Len':<10} | {'Max Grad Norm':<14}")
    print("-" * 75)
    for k, s in fork_summaries.items():
        print(f"{s['kl_beta']:<8.2f} | {s['mean_reward']:<12.3f} | {s['final_kl']:<10.4f} | {s['mean_entropy']:<12.3f} | {s['mean_response_length']:<10.1f} | {s['max_grad_norm']:<14.3f}")
    print(f"\nSaved ablation summary to: {summary_file}")
    return fork_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    run_kl_ablation(args.config)


if __name__ == "__main__":
    main()
