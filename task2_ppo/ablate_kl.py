from __future__ import annotations

import argparse
from pathlib import Path
from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import run_evaluation


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
        train_summary = run_ppo(
            config_path=config_path,
            output=out_dir,
            updates=fork_updates,
            kl_beta=beta,
            run_name=run_name,
        )
        heldout = run_evaluation(
            config_path=config_path,
            adapter=out_dir,
            name=run_name,
        )
        fork_summaries[f"beta_{beta}"] = {
            "kl_beta": beta,
            "training": train_summary,
            "heldout_reward": heldout["reward_mean"],
            "heldout_kl": heldout["kl_mean"],
            "heldout_entropy": heldout["entropy_mean"],
            "heldout_response_length": heldout["response_length_mean"],
            "heldout_truncation_rate": heldout["truncation_rate_at_cap"],
        }

    summary_file = results_dir / "kl_ablation_summary.json"
    save_json(summary_file, fork_summaries)

    print("\n=== KL Beta Ablation Summary ===")
    print(f"{'Beta':<8} | {'Held-out Reward':<16} | {'Held-out KL':<12} | {'Held-out Entropy':<18} | {'Held-out Length':<17}")
    print("-" * 96)
    for k, s in fork_summaries.items():
        print(f"{s['kl_beta']:<8.2f} | {s['heldout_reward']:<16.3f} | {s['heldout_kl']:<12.4f} | {s['heldout_entropy']:<18.3f} | {s['heldout_response_length']:<17.1f}")
    print(f"\nSaved ablation summary to: {summary_file}")
    return fork_summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    run_kl_ablation(args.config)


if __name__ == "__main__":
    main()
