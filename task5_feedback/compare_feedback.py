from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


def task5_results_dir(cfg: dict):
    return repo_path(cfg.get("results_dir", "results")) / "task5_feedback"


def run_comparison(cfg: dict) -> dict:
    outdir = task5_results_dir(cfg)
    required = {
        "gsm": outdir / "gsm_summary.json",
        "transfer": outdir / "transfer_summary.json",
        "diagnostic": outdir / "diagnostic_summary.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Run the GSM, transfer, and diagnostic evaluations first:\n" + "\n".join(missing))

    gsm = load_json(required["gsm"])
    transfer = load_json(required["transfer"])
    diagnostic = load_json(required["diagnostic"])
    records = []
    for policy_name in ("sft", "rlvr", "rlaif"):
        in_domain = gsm["policies"][policy_name]
        transfer_metrics = transfer["policies"][policy_name]
        record = {
            "policy": policy_name,
            "gsm_exact_accuracy": in_domain["exact_accuracy"],
            "gsm_format_compliance": in_domain["format_compliance"],
            "gsm_mean_response_length": in_domain["mean_response_length"],
            "transfer_exact_accuracy": transfer_metrics["exact_accuracy"],
            "transfer_format_compliance": transfer_metrics["format_compliance"],
            "transfer_mean_response_length": transfer_metrics["mean_response_length"],
            "exact_accuracy_drop": in_domain["exact_accuracy"] - transfer_metrics["exact_accuracy"],
        }
        if policy_name != "sft":
            record.update({
                "gsm_ai_win_rate_vs_sft": gsm["pairwise_vs_sft"][policy_name]["ai_win_rate_vs_sft"],
                "gsm_verifier_judge_agreement": gsm["pairwise_vs_sft"][policy_name]["verifier_judge_agreement"],
                "transfer_ai_win_rate_vs_sft": transfer["pairwise_vs_sft"][policy_name]["ai_win_rate_vs_sft"],
                "transfer_verifier_judge_agreement": transfer["pairwise_vs_sft"][policy_name]["verifier_judge_agreement"],
            })
        records.append(record)

    summary = {
        "in_domain": gsm,
        "transfer": transfer,
        "diagnostic": diagnostic,
        "policy_comparison": records,
    }
    save_json(outdir / "feedback_comparison_summary.json", summary)
    pd.DataFrame(records).to_csv(outdir / "feedback_policy_comparison.csv", index=False)
    print(f"Saved comparison summary to: {outdir / 'feedback_comparison_summary.json'}")
    print(f"Saved policy table to: {outdir / 'feedback_policy_comparison.csv'}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    summary = run_comparison(load_yaml(args.config))
    for row in summary["policy_comparison"]:
        print(f"{row['policy']}: GSM exact={row['gsm_exact_accuracy']:.3f}, transfer exact={row['transfer_exact_accuracy']:.3f}, drop={row['exact_accuracy_drop']:.3f}")


if __name__ == "__main__":
    main()
