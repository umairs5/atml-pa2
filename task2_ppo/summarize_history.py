from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task2_ppo.continue_train import summarize_ppo_history


def main():
    ap = argparse.ArgumentParser(description="Save a PPO summary from an existing training history JSONL.")
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--history", required=True, help="Path to an existing PPO training history JSONL.")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--adapter", help="Optional adapter path recorded in the summary.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    records = read_jsonl(args.history)
    summary = summarize_ppo_history(records, run_name=args.name, adapter_path=args.adapter)
    outdir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{args.name}_train_summary.json"
    save_json(path, summary)
    print(f"Saved training summary to: {path}")


if __name__ == "__main__":
    main()
