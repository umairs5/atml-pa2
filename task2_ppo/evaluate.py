from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 6304)))
    rm_model, rm_tok = load_reward_model(cfg)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward_model": rm_model,
        "reward_tokenizer": rm_tok,
    }


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str = "standard",
    max_examples: int | None = None,
    batch_size: int = 2,
):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    rows = bundle["rows"]
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    rm_model = bundle["reward_model"]
    rm_tok = bundle["reward_tokenizer"]

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_new_tokens = int(cfg.get("eval_max_response_length", 768))
    gen_cfg = cfg.get("generation", {})

    print(f"\n--- Evaluating PPO policy '{name}' on {len(rows)} held-out prompts ---")

    gen_records = []
    rewards = []
    kl_values = []
    lengths = []
    eos_terminated = []
    truncated_flags = []

    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        pm_list = [prompt_messages(r) for r in chunk]

        with torch.no_grad():
            gen_out = batch_generate(
                model=policy,
                tokenizer=tokenizer,
                prompts=pm_list,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                temperature=float(gen_cfg.get("temperature", 0.7)),
                top_p=float(gen_cfg.get("top_p", 0.9)),
                do_sample=bool(gen_cfg.get("do_sample", True)),
            )

            seqs = gen_out["sequences"]
            attn = gen_out["attention_mask"]
            pw = gen_out["prompt_width"]
            rids = gen_out["response_ids"]
            rmask = gen_out["response_mask"]

            pol_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)

            rm_scores = score_reward_pairs(rm_model, rm_tok, pm_list, gen_out["responses"])

            for b in range(len(chunk)):
                m = rmask[b]
                kl = sampled_kl(pol_logp[b], ref_logp[b], m).item()
                r_val = float(rm_scores[b].item())
                l_val = int(m.sum().item())
                is_eos = gen_out["terminated_with_eos"][b]
                is_trunc = gen_out["truncated"][b]

                rewards.append(r_val)
                kl_values.append(kl)
                lengths.append(l_val)
                eos_terminated.append(is_eos)
                truncated_flags.append(is_trunc)

                gen_records.append({
                    "source_index": chunk[b].get("source_index"),
                    "prompt_id": chunk[b].get("prompt_id"),
                    "prompt": chunk[b].get("prompt"),
                    "response": gen_out["responses"][b],
                    "reward": r_val,
                    "sampled_kl": kl,
                    "response_length": l_val,
                    "terminated_with_eos": is_eos,
                    "truncated": is_trunc,
                })

        if (i // batch_size + 1) % 5 == 0 or (i + batch_size >= len(rows)):
            print(f"Processed {min(i + batch_size, len(rows))}/{len(rows)} prompts...")

    len_mean = float(np.mean(lengths)) if lengths else 0.0
    len_std = float(np.std(lengths)) if lengths else 0.0
    len_iqr = float(np.percentile(lengths, 75) - np.percentile(lengths, 25)) if lengths else 0.0

    rew_mean = float(np.mean(rewards)) if rewards else 0.0
    rew_std = float(np.std(rewards)) if rewards else 0.0

    kl_mean = float(np.mean(kl_values)) if kl_values else 0.0
    kl_std = float(np.std(kl_values)) if kl_values else 0.0

    eos_rate = float(np.mean(eos_terminated)) if eos_terminated else 0.0
    trunc_rate = float(np.mean(truncated_flags)) if truncated_flags else 0.0

    summary = {
        "name": name,
        "adapter_path": adapter,
        "num_eval_prompts": len(rows),
        "reward_mean": rew_mean,
        "reward_std": rew_std,
        "kl_mean": kl_mean,
        "kl_std": kl_std,
        "response_length_mean": len_mean,
        "response_length_std": len_std,
        "response_length_iqr": len_iqr,
        "eos_termination_rate": eos_rate,
        "truncation_rate_at_cap": trunc_rate,
    }

    summary_path = results_dir / f"{name}_eval_summary.json"
    save_json(summary_path, summary)

    generations_path = results_dir / f"{name}_eval_generations.jsonl"
    write_jsonl(generations_path, gen_records)

    print(f"\n=== Evaluation Results for '{name}' ===")
    print(f"  Reward              : {rew_mean:.3f} +/- {rew_std:.3f}")
    print(f"  Reference KL        : {kl_mean:.4f} +/- {kl_std:.4f}")
    print(f"  Response Length     : {len_mean:.1f} +/- {len_std:.1f} (IQR={len_iqr:.1f})")
    print(f"  EOS Termination Rate: {eos_rate * 100:.1f}%")
    print(f"  Truncation Rate     : {trunc_rate * 100:.1f}%")
    print(f"Saved evaluation summary to: {summary_path}")
    print(f"Saved generations to: {generations_path}")

    clear_gpu(policy, rm_model)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    run_evaluation(
        config_path=args.config,
        adapter=args.adapter,
        name=args.name,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
