from __future__ import annotations

import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
)
from common.logging_utils import save_json, set_seed
from common.metrics import word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import run_training


def evaluate_stratified_preference(cfg: dict, adapter_path: str, eval_path: str, max_examples: int | None = None):
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
    rows = read_jsonl(eval_path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    device = next(policy.parameters()).device
    beta = float(cfg.get("beta", 0.10))
    max_seq_len = int(cfg.get("max_sequence_length", 768))

    strata_margins = defaultdict(list)

    eval_batch_size = int(cfg.get("batch_size", 2))
    for i in range(0, len(rows), eval_batch_size):
        chunk = rows[i : i + eval_batch_size]
        chosen_enc, rejected_enc = [], []
        for r in chunk:
            pm = prompt_messages_from_preference(r)
            yc, yr = preference_responses(r)
            chosen_enc.append(encode_prompt_response(tokenizer, pm, yc, max_seq_len))
            rejected_enc.append(encode_prompt_response(tokenizer, pm, yr, max_seq_len))

        cb = {k: v.to(device) for k, v in pad_batch(tokenizer, chosen_enc).items()}
        rb = {k: v.to(device) for k, v in pad_batch(tokenizer, rejected_enc).items()}

        with torch.no_grad():
            with reference_mode(policy):
                ref_c_logp, _, _ = response_sequence_logprobs(policy, cb)
                ref_r_logp, _, _ = response_sequence_logprobs(policy, rb)
            pol_c_logp, _, _ = response_sequence_logprobs(policy, cb)
            pol_r_logp, _, _ = response_sequence_logprobs(policy, rb)

        implicit_margin = (pol_c_logp - pol_r_logp) - (ref_c_logp - ref_r_logp)
        for j, r in enumerate(chunk):
            stratum = r.get("length_stratum", "unknown")
            strata_margins[stratum].append(float(implicit_margin[j].item()))

    strata_acc = {}
    for stratum, margins in strata_margins.items():
        acc = float(np.mean([m > 0 for m in margins])) if margins else 0.0
        strata_acc[stratum] = {
            "count": len(margins),
            "preference_accuracy": acc,
            "mean_implicit_margin": float(np.mean(margins)) if margins else 0.0,
        }

    all_m = [m for sublist in strata_margins.values() for m in sublist]
    strata_acc["overall"] = {
        "count": len(all_m),
        "preference_accuracy": float(np.mean([m > 0 for m in all_m])) if all_m else 0.0,
        "mean_implicit_margin": float(np.mean(all_m)) if all_m else 0.0,
    }

    clear_gpu(policy)
    return strata_acc


def evaluate_word_limit(cfg: dict, adapter_path: str, prompt_path: str):
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
    rows = read_jsonl(prompt_path)
    prompts = [prompt_messages(r) for r in rows]

    gen_cfg = cfg.get("generation", {})
    gen_out = batch_generate(
        model=policy,
        tokenizer=tokenizer,
        prompts=prompts,
        max_prompt_length=int(cfg.get("max_sequence_length", 768)),
        max_new_tokens=int(cfg.get("max_generation_tokens", 256)),
        temperature=float(gen_cfg.get("temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.9)),
        do_sample=False,  # Deterministic generation for fair instruction-following evaluation
    )

    records = []
    compliances = []
    word_counts = []

    for i, r in enumerate(rows):
        prompt_str = r.get("prompt", "")
        if not prompt_str and "messages" in r:
            prompt_str = r["messages"][0]["content"]

        resp_text = gen_out["responses"][i]
        w_count = word_count(resp_text)
        compliant = word_limit_compliance(prompt_str, resp_text)

        compliances.append(compliant)
        word_counts.append(w_count)
        records.append({
            "prompt_id": r.get("prompt_id"),
            "prompt": prompt_str,
            "response": resp_text,
            "word_count": w_count,
            "compliant": compliant,
        })

    clear_gpu(policy)
    valid_comp = [c for c in compliances if c is not None]
    return {
        "word_limit_compliance_rate": float(np.mean(valid_comp)) if valid_comp else 0.0,
        "mean_word_count": float(np.mean(word_counts)) if word_counts else 0.0,
        "std_word_count": float(np.std(word_counts)) if word_counts else 0.0,
        "records": records,
    }


def run_length_study(
    config_path: str,
    train_balanced: bool = True,
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
):
    cfg = load_yaml(config_path)
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    std_adapter = str(repo_path(cfg["standard_output"]))
    len_adapter = str(repo_path(cfg["length_output"]))

    if train_balanced:
        print("\n=== Training Length-Balanced DPO Model ===")
        run_training(
            config_path=config_path,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=cfg["length_output"],
            max_examples=max_train_examples,
        )

    # Verify standard adapter exists
    if not repo_path(std_adapter).exists():
        print(f"Standard DPO adapter not found at {std_adapter}. Training standard model first...")
        run_training(
            config_path=config_path,
            run_name="standard",
            dataset_path=cfg["paths"]["dpo_standard_train"],
            output_path=cfg["standard_output"],
            max_examples=max_train_examples,
        )

    print("\n=== Evaluating Stratified Preference Accuracy ===")
    stratified_path = cfg["paths"]["dpo_length_eval"]

    print("Evaluating Standard DPO on stratified eval set...")
    std_stratified = evaluate_stratified_preference(cfg, std_adapter, stratified_path, max_eval_examples)

    print("Evaluating Length-Balanced DPO on stratified eval set...")
    len_stratified = evaluate_stratified_preference(cfg, len_adapter, stratified_path, max_eval_examples)

    print("\n=== Evaluating Word-Limit Compliance ===")
    word_limit_path = cfg["paths"]["word_limit_prompts"]

    print("Evaluating Standard DPO on word-limit prompts...")
    std_word_eval = evaluate_word_limit(cfg, std_adapter, word_limit_path)

    print("Evaluating Length-Balanced DPO on word-limit prompts...")
    len_word_eval = evaluate_word_limit(cfg, len_adapter, word_limit_path)

    summary = {
        "standard": {
            "stratified_preference": std_stratified,
            "word_limit": {
                "compliance_rate": std_word_eval["word_limit_compliance_rate"],
                "mean_word_count": std_word_eval["mean_word_count"],
                "std_word_count": std_word_eval["std_word_count"],
            },
        },
        "length_balanced": {
            "stratified_preference": len_stratified,
            "word_limit": {
                "compliance_rate": len_word_eval["word_limit_compliance_rate"],
                "mean_word_count": len_word_eval["mean_word_count"],
                "std_word_count": len_word_eval["std_word_count"],
            },
        },
    }

    summary_file = results_dir / "length_confounding_summary.json"
    save_json(summary_file, summary)

    # Save detailed word-limit generations
    write_jsonl(results_dir / "standard_word_limit_generations.jsonl", std_word_eval["records"])
    write_jsonl(results_dir / "length_balanced_word_limit_generations.jsonl", len_word_eval["records"])

    # Pretty print comparison
    print("\n" + "=" * 65)
    print("LENGTH-CONFOUNDING STUDY SUMMARY")
    print("=" * 65)
    strata = ["preferred_longer", "length_matched", "rejected_longer", "overall"]
    table_rows = []
    for s in strata:
        std_p = std_stratified.get(s, {}).get("preference_accuracy", 0.0)
        len_p = len_stratified.get(s, {}).get("preference_accuracy", 0.0)
        table_rows.append({
            "Stratum": s,
            "Standard DPO Acc": f"{std_p:.4f}",
            "Balanced DPO Acc": f"{len_p:.4f}",
            "Delta": f"{len_p - std_p:+.4f}",
        })
    df_strata = pd.DataFrame(table_rows)
    print(df_strata.to_string(index=False))

    print("\n" + "-" * 65)
    print("WORD-LIMIT COMPLIANCE COMPARISON")
    print("-" * 65)
    print(f"Standard DPO        : Compliance = {std_word_eval['word_limit_compliance_rate']*100:.1f}%, Mean Words = {std_word_eval['mean_word_count']:.1f}")
    print(f"Length-Balanced DPO : Compliance = {len_word_eval['word_limit_compliance_rate']*100:.1f}%, Mean Words = {len_word_eval['mean_word_count']:.1f}")
    print(f"Saved complete length study summary to {summary_file}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--no-train", action="store_true", help="Skip training length-balanced model if already trained")
    ap.add_argument("--max-train-examples", type=int, help="Override training examples for smoke-testing")
    ap.add_argument("--max-eval-examples", type=int, help="Override eval examples for smoke-testing")
    args = ap.parse_args()
    run_length_study(
        config_path=args.config,
        train_balanced=not args.no_train,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
    )


if __name__ == "__main__":
    main()
