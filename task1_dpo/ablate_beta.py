from __future__ import annotations

import argparse
import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training


def run_beta_ablation(config_path: str, max_examples: int | None = None, skip_eval: bool = False):
    cfg = load_yaml(config_path)
    betas = cfg.get("betas", [0.03, 0.10, 0.30])
    short_examples = max_examples or int(cfg.get("short_ablation_examples", 600))
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting DPO Beta Regularization Study ===")
    print(f"Beta sweep values: {betas}")
    print(f"Budget per condition: {short_examples} examples (from original policy initialization)")

    summaries = []
    for beta in betas:
        run_name = f"beta_{beta}"
        output_dir = f"outputs/task1_dpo/{run_name}"
        print(f"\n=======================================================")
        print(f"Running condition: beta = {beta} (name: {run_name})")
        print(f"=======================================================")

        run_training(
            config_path=config_path,
            run_name=run_name,
            beta=beta,
            max_examples=short_examples,
            output_path=output_dir,
        )

        if not skip_eval:
            eval_summary = run_evaluation(
                config_path=config_path,
                adapter=output_dir,
                name=run_name,
            )
            eval_summary["beta"] = beta
            eval_summary["budget_examples"] = short_examples
            summaries.append(eval_summary)

    if summaries:
        summary_file = results_dir / "beta_ablation_summary.json"
        csv_file = results_dir / "beta_ablation_summary.csv"
        save_json(summary_file, summaries)

        df = pd.DataFrame(summaries)
        display_cols = [
            "beta",
            "preference_accuracy",
            "dpo_loss",
            "kl_mean",
            "reward_score_mean",
            "response_length_mean",
            "response_length_std",
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        df[available_cols].to_csv(csv_file, index=False)

        print("\n=== Beta Ablation Summary Table ===")
        print(df[available_cols].to_string(index=False))
        print(f"\nSaved beta summary to {summary_file} and {csv_file}")

    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--max-examples", type=int, help="Override short_ablation_examples for smoke-testing")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    run_beta_ablation(args.config, max_examples=args.max_examples, skip_eval=args.skip_eval)


if __name__ == "__main__":
    main()
