from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def make_audit_sheet(cfg: dict, overwrite: bool = False):
    """Create a policy-blinded manual labeling sheet for the fixed 60 prompts."""
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    source = outdir / "generated_sft.jsonl"
    if not source.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(source))

    ids_path = outdir / "manual_audit_ids.csv"
    template_path = outdir / "manual_audit_template.csv"
    key_path = outdir / "manual_audit_key.csv"
    if not overwrite and template_path.exists() and key_path.exists():
        print(f"Using existing blinded audit sheet: {template_path}")
        return template_path

    base_rows = read_jsonl(source)
    ids = fixed_audit_ids(base_rows, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids}).to_csv(ids_path, index=False)

    audit_rows, key_rows = [], []
    row_number = 1
    for policy in ["sft", "dpo", "ppo", "grpo"]:
        response_path = outdir / f"generated_{policy}.jsonl"
        if not response_path.exists():
            raise FileNotFoundError(f"Generate/save responses first: {response_path}")
        by_id = {int(row["xstest_id"]): row for row in read_jsonl(response_path)}
        missing = sorted(set(ids) - set(by_id))
        if missing:
            raise ValueError(f"{response_path} is missing audit IDs: {missing[:8]}")
        for xstest_id in ids:
            row = by_id[xstest_id]
            audit_id = f"audit_{row_number:03d}"
            audit_rows.append({
                "audit_row_id": audit_id,
                "xstest_id": xstest_id,
                "benchmark_class": row["benchmark_class"],
                "type": row["type"],
                "prompt": row["prompt"],
                "response": row["response"],
                "manual_label": "",
            })
            key_rows.append({"audit_row_id": audit_id, "policy": policy, "xstest_id": xstest_id})
            row_number += 1

    # A fixed shuffle conceals policy order without exposing AI judge labels.
    rng = np.random.default_rng(int(cfg["seed"]))
    order = rng.permutation(len(audit_rows))
    pd.DataFrame([audit_rows[i] for i in order]).to_csv(template_path, index=False)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)
    print(f"Wrote fixed audit IDs: {ids_path}")
    print(f"Wrote blinded manual-label template: {template_path}")
    print(f"Keep the policy key separate until labels are complete: {key_path}")
    return template_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    make_audit_sheet(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
