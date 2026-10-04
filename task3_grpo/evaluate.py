from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import sample_entropy, sampled_kl
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

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_new_tokens = int(cfg.get("eval_max_response_length", 768))

    print(f"\n--- Evaluating GRPO policy '{name}' on {len(rows)} held-out prompts ---")

    gen_records = []
    rewards = []
    kl_values = []
    entropies = []
    lengths = []
    terminated_eos_list = []

    policy.eval()

    for idx in range(0, len(rows), batch_size):
        batch_rows = rows[idx : idx + batch_size]
        pm_list = [prompt_messages(r) for r in batch_rows]

        with torch.no_grad():
            gen_out = batch_generate(
                policy,
                tokenizer,
                pm_list,
                max_prompt_length=max_prompt_length,
                max_response_length=max_new_tokens,
            )

            seqs = gen_out["sequences"]
            attn = gen_out["attention_mask"]
            pw = gen_out["prompt_width"]
            rids = gen_out["response_ids"]
            rmask = gen_out["response_mask"]
            responses = gen_out["responses"]
            eos_flags = gen_out["terminated_with_eos"]

            scores = score_reward_pairs(rm_model, rm_tok, pm_list, responses)

            pol_lp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)
            with reference_mode(policy):
                ref_lp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)

            for b in range(len(batch_rows)):
                m = rmask[b]
                num_tokens = int(m.sum().item())
                r_score = float(scores[b])
                eos = bool(eos_flags[b])

                p_lp_b = pol_lp[b : b + 1]
                r_lp_b = ref_lp[b : b + 1]
                m_b = m.unsqueeze(0)

                kl_val = float(sampled_kl(p_lp_b, r_lp_b, m_b).item())
                ent_val = float(sample_entropy(p_lp_b, m_b).item())

                rewards.append(r_score)
                kl_values.append(kl_val)
                entropies.append(ent_val)
                lengths.append(num_tokens)
                terminated_eos_list.append(eos)

                gen_records.append({
                    "prompt_id": batch_rows[b].get("prompt_id", str(idx + b)),
                    "completion": responses[b],
                    "reward": r_score,
                    "kl": kl_val,
                    "entropy": ent_val,
                    "response_length": num_tokens,
                    "terminated_with_eos": eos,
                    "truncated": not eos,
                })

        if (idx + batch_size) % 10 == 0 or (idx + len(batch_rows)) == len(rows):
            print(f"Processed {min(idx + batch_size, len(rows))}/{len(rows)} prompts...")

    lengths_arr = np.array(lengths)
    q25, q75 = np.percentile(lengths_arr, [25, 75])
    iqr = float(q75 - q25)
    eos_rate = float(np.mean(terminated_eos_list))
    truncation_rate = 1.0 - eos_rate

    summary = {
        "name": name,
        "adapter_path": adapter,
        "num_eval_prompts": len(rows),
        "reward_mean": round(float(np.mean(rewards)), 3),
        "reward_std": round(float(np.std(rewards)), 3),
        "kl_mean": round(float(np.mean(kl_values)), 4),
        "kl_std": round(float(np.std(kl_values)), 4),
        "response_length_mean": round(float(np.mean(lengths_arr)), 1),
        "response_length_std": round(float(np.std(lengths_arr)), 1),
        "response_length_iqr": round(iqr, 1),
        "eos_termination_rate": round(eos_rate, 3),
        "truncation_rate_at_cap": round(truncation_rate, 3),
    }

    print(f"\n=== Evaluation Results for '{name}' ===")
    print(f"  Reward              : {summary['reward_mean']} +/- {summary['reward_std']}")
    print(f"  Reference KL        : {summary['kl_mean']} +/- {summary['kl_std']}")
    print(f"  Response Length     : {summary['response_length_mean']} +/- {summary['response_length_std']} (IQR={summary['response_length_iqr']})")
    print(f"  EOS Termination Rate: {summary['eos_termination_rate']*100:.1f}%")
    print(f"  Truncation Rate     : {summary['truncation_rate_at_cap']*100:.1f}%")

    out_file = results_dir / f"{name}_eval_summary.json"
    save_json(out_file, summary)
    print(f"Saved evaluation summary to: {out_file}")

    gen_file = results_dir / f"{name}_eval_generations.jsonl"
    write_jsonl(gen_file, gen_records)
    print(f"Saved generations to: {gen_file}")

    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
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
