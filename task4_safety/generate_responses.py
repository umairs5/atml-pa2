from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    clear_gpu(model)
    return records


def task4_results_dir(cfg) -> Path:
    return repo_path(cfg["results_dir"]) / "task4_safety"


def generate_all_policies(
    cfg: dict,
    batch_size: int = 4,
    policies: list[str] | None = None,
    overwrite: bool = False,
):
    """Generate one deterministic response per fixed policy/prompt pair."""
    selected = policies or list(policy_specs(cfg))
    unknown = set(selected) - set(policy_specs(cfg))
    if unknown:
        raise ValueError(f"Unknown policies: {sorted(unknown)}")

    outdir = task4_results_dir(cfg)
    outdir.mkdir(parents=True, exist_ok=True)
    expected_rows = len(load_xstest(cfg))
    outputs = {}
    for policy_name in selected:
        output = outdir / f"generated_{policy_name}.jsonl"
        if output.exists() and not overwrite:
            existing = sum(1 for line in output.open(encoding="utf-8") if line.strip())
            if existing == expected_rows:
                print(f"Using existing deterministic generations: {output}")
                outputs[policy_name] = output
                continue
            raise RuntimeError(
                f"{output} contains {existing} rows; expected {expected_rows}. "
                "Use --overwrite after checking the file."
            )

        # Generation is deterministic, but fixing the seed protects reproducibility
        # should a future generation setting be changed.
        set_seed(int(cfg["seed"]))
        print(f"Generating deterministic responses for: {policy_name}")
        records = generate_for_policy(cfg, policy_name, batch_size=batch_size)
        if len(records) != expected_rows:
            raise RuntimeError(f"Generated {len(records)} rows for {policy_name}; expected {expected_rows}")
        write_jsonl(output, records)
        print(f"Saved {len(records)} responses to: {output}")
        outputs[policy_name] = output
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--policies", nargs="+", choices=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outputs = generate_all_policies(
        cfg,
        batch_size=args.batch_size,
        policies=args.policies,
        overwrite=args.overwrite,
    )
    print("Generated files:", {name: str(path) for name, path in outputs.items()})


if __name__ == "__main__":
    main()
