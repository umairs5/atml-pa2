from __future__ import annotations

import argparse
import numpy as np
import torch
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss


def load_evaluation_bundle(config_path: str, adapter: str, dataset_path: str | None = None, load_rm: bool = True):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_eval"]
    rows = read_jsonl(path)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm_model, rm_tok = load_reward_model(cfg) if load_rm else (None, None)
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": rm_model,
        "reward_tokenizer": rm_tok,
    }


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str = "standard",
    dataset_path: str | None = None,
    max_examples: int | None = None,
    load_rm: bool = True,
):
    bundle = load_evaluation_bundle(config_path, adapter, dataset_path=dataset_path, load_rm=load_rm)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    rm_model = bundle["reward_model"]
    rm_tok = bundle["reward_tokenizer"]
    beta = float(cfg.get("beta", 0.10))
    max_seq_len = int(cfg.get("max_sequence_length", 768))
    device = next(policy.parameters()).device

    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Evaluating DPO model '{name}' on {len(rows)} examples ---")

    # 1. Held-out Preference Evaluation
    eval_batch_size = int(cfg.get("batch_size", 2))
    all_margins = []
    all_losses = []
    pair_records = []

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
                ref_chosen_logp, _, _ = response_sequence_logprobs(policy, cb)
                ref_rejected_logp, _, _ = response_sequence_logprobs(policy, rb)

            pol_chosen_logp, _, _ = response_sequence_logprobs(policy, cb)
            pol_rejected_logp, _, _ = response_sequence_logprobs(policy, rb)

            loss, diag = dpo_loss(pol_chosen_logp, pol_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta)

        implicit_margin = (pol_chosen_logp - pol_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
        for j, r in enumerate(chunk):
            m_val = float(implicit_margin[j].item())
            all_margins.append(m_val)
            all_losses.append(float(loss.item()))
            pair_records.append({
                "prompt_id": r.get("prompt_id"),
                "source_index": r.get("source_index"),
                "implicit_margin": m_val,
                "preferred_correct": bool(m_val > 0),
            })

    pref_acc = float(np.mean([m > 0 for m in all_margins])) if all_margins else 0.0
    mean_dpo_loss = float(np.mean(all_losses)) if all_losses else 0.0

    # 2. Generation & Reference KL + Reward Model Scoring
    gen_records = []
    gen_lengths = []
    kl_values = []
    reward_scores = []

    gen_cfg = cfg.get("generation", {})
    gen_batch_size = int(cfg.get("batch_size", 2))
    max_new_tokens = int(cfg.get("max_generation_tokens", 256))

    for i in range(0, len(rows), gen_batch_size):
        chunk = rows[i : i + gen_batch_size]
        prompts = [prompt_messages_from_preference(r) for r in chunk]

        with torch.no_grad():
            gen_out = batch_generate(
                model=policy,
                tokenizer=tokenizer,
                prompts=prompts,
                max_prompt_length=max_seq_len,
                max_new_tokens=max_new_tokens,
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                do_sample=bool(gen_cfg.get("do_sample", True)),
            )

            # Compute KL on generated tokens
            seqs = gen_out["sequences"]
            attn = gen_out["attention_mask"]
            pw = gen_out["prompt_width"]
            rids = gen_out["response_ids"]
            rmask = gen_out["response_mask"]

            pol_tok_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)
            with reference_mode(policy):
                ref_tok_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)

            for b in range(len(chunk)):
                m = rmask[b]
                kl = sampled_kl(pol_tok_logp[b], ref_tok_logp[b], m).item()
                kl_values.append(float(kl))
                gen_lengths.append(int(m.sum().item()))

            # Reward model scoring
            if rm_model is not None and rm_tok is not None:
                scores = score_reward_pairs(rm_model, rm_tok, prompts, gen_out["responses"])
                for b in range(len(chunk)):
                    reward_scores.append(float(scores[b].item()))

        for b, r in enumerate(chunk):
            rec = {
                "prompt_id": r.get("prompt_id"),
                "prompt": r.get("prompt"),
                "generated_response": gen_out["responses"][b],
                "response_length": int(rmask[b].sum().item()),
                "sampled_kl": kl_values[-len(chunk) + b],
            }
            if reward_scores:
                rec["reward_score"] = reward_scores[-len(chunk) + b]
            gen_records.append(rec)

    # Calculate statistics
    len_mean = float(np.mean(gen_lengths)) if gen_lengths else 0.0
    len_std = float(np.std(gen_lengths)) if gen_lengths else 0.0
    len_iqr = float(np.percentile(gen_lengths, 75) - np.percentile(gen_lengths, 25)) if gen_lengths else 0.0

    kl_mean = float(np.mean(kl_values)) if kl_values else 0.0
    kl_std = float(np.std(kl_values)) if kl_values else 0.0

    rm_mean = float(np.mean(reward_scores)) if reward_scores else None
    rm_std = float(np.std(reward_scores)) if reward_scores else None

    summary = {
        "name": name,
        "adapter_path": adapter,
        "num_eval_examples": len(rows),
        "dpo_loss": mean_dpo_loss,
        "preference_accuracy": pref_acc,
        "mean_implicit_margin": float(np.mean(all_margins)) if all_margins else 0.0,
        "kl_mean": kl_mean,
        "kl_std": kl_std,
        "reward_score_mean": rm_mean,
        "reward_score_std": rm_std,
        "response_length_mean": len_mean,
        "response_length_std": len_std,
        "response_length_iqr": len_iqr,
    }

    summary_path = results_dir / f"{name}_eval_summary.json"
    save_json(summary_path, summary)

    generations_path = results_dir / f"{name}_eval_generations.jsonl"
    write_jsonl(generations_path, gen_records)

    print(f"Results for '{name}':")
    print(f"  Preference Accuracy : {pref_acc:.4f}")
    print(f"  DPO Loss            : {mean_dpo_loss:.4f}")
    print(f"  Reference KL        : {kl_mean:.4f} +/- {kl_std:.4f}")
    if rm_mean is not None:
        print(f"  Reward Score        : {rm_mean:.4f} +/- {rm_std:.4f}")
    print(f"  Response Length     : {len_mean:.1f} +/- {len_std:.1f} (IQR={len_iqr:.1f})")
    print(f"Saved evaluation summary to: {summary_path}")

    clear_gpu(policy, rm_model)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--skip-rm", action="store_true")
    args = ap.parse_args()
    run_evaluation(
        config_path=args.config,
        adapter=args.adapter,
        name=args.name,
        dataset_path=args.dataset,
        max_examples=args.max_examples,
        load_rm=not args.skip_rm,
    )


if __name__ == "__main__":
    main()
