from __future__ import annotations

import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, mean_response_length, sample_entropy, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    
    out = repo_path(output or cfg.get("output", f"outputs/task3_grpo/{run_name}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    num_updates = int(cfg.get("updates", 20))
    clip_eps = float(cfg.get("clip_epsilon", 0.20))
    kl_beta = float(cfg.get("kl_beta", 0.10))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    num_generations = int(cfg.get("num_generations", 4))
    policy_epochs = int(cfg.get("policy_epochs", 1))
    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_completion_length = int(cfg.get("max_completion_length", 512))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    tokenizer = bundle["tokenizer"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    device = next(policy.parameters()).device
    timer = wall_timer()
    history_file = out.parent / f"{run_name}_train_history.jsonl"
    if history_file.exists():
        history_file.unlink()

    print(f"\n=== Starting GRPO Run: {run_name} ===")
    print(f"Updates: {num_updates} | Loss Type: {loss_type} | K (num_generations): {num_generations}")
    print(f"Prompts/update: {prompts_per_update} | Clip eps: {clip_eps} | KL beta: {kl_beta}")
    print(f"Mask truncated completions: {mask_truncated} | Output: {out}\n")

    history_records = []

    for step in range(1, num_updates + 1):
        policy.eval()

        # Sample prompts
        batch_prompts = random.choices(prompts, k=prompts_per_update)
        
        # Build prompt list with K completions per prompt
        pm_list = []
        group_ids_list = []
        for gid, p in enumerate(batch_prompts):
            msgs = prompt_messages(p)
            for _ in range(num_generations):
                pm_list.append(msgs)
                group_ids_list.append(gid)

        group_ids = torch.tensor(group_ids_list, device=device, dtype=torch.long)

        with torch.no_grad():
            gen_out = batch_generate(
                policy,
                tokenizer,
                pm_list,
                max_prompt_length=max_prompt_length,
                max_response_length=max_completion_length,
            )
            seqs = gen_out["sequences"].clone()
            attn = gen_out["attention_mask"].clone()
            pw = gen_out["prompt_width"]
            rids = gen_out["response_ids"].clone()
            rmask = gen_out["response_mask"].clone()
            responses = gen_out["responses"]
            terminated_eos = gen_out["terminated_with_eos"]

            # Mask completions that hit the maximum generation length without EOS
            if mask_truncated:
                truncated = [not eos for eos in terminated_eos]
                rmask = mask_truncated_sequences(rmask, truncated)

            # Score generated responses with reward model
            rm_scores = score_reward_pairs(reward_model, reward_tokenizer, pm_list, responses)
            rewards = torch.as_tensor(rm_scores, device=device, dtype=torch.float32)

            # Compute group-relative advantages
            advantages = group_relative_advantages(rewards, group_ids)

            # Within-group statistics
            group_stds = []
            uninformative_count = 0
            for gid in torch.unique(group_ids):
                g_rewards = rewards[group_ids == gid]
                g_std = float(g_rewards.std(unbiased=False).item())
                group_stds.append(g_std)
                if g_std <= 1e-6:
                    uninformative_count += 1
            
            mean_group_std = float(np.mean(group_stds)) if group_stds else 0.0
            uninformative_frac = float(uninformative_count / len(group_stds)) if group_stds else 0.0

            # Log probabilities under current rollout policy
            old_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)

            # Reference model log probabilities
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)

        # Policy optimization step
        policy.train()
        for _ in range(policy_epochs):
            new_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)
            loss, diag = grpo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                seq_adv=advantages,
                token_mask=rmask,
                ref_logp=ref_logp,
                eps=clip_eps,
                beta=kl_beta,
                loss_type=loss_type,
                max_completion_length=max_completion_length,
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                optimizer.step()
            else:
                print("WARNING: NaN or Inf gradient in GRPO policy! Skipping step.")
                optimizer.zero_grad()

        # Step metrics
        mean_rew = float(rewards.mean().item())
        mean_kl = float(diag["sampled_kl"].item())
        mean_ent = float(diag["sample_entropy"].item())
        mean_len = mean_response_length(rmask)
        clip_frac = float(diag["clip_fraction"].item())
        p_loss_val = float(diag["policy_term"].item())
        gnorm_val = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)

        record = {
            "step": step,
            "reward_mean": round(mean_rew, 4),
            "reward_group_std": round(mean_group_std, 4),
            "uninformative_fraction": round(uninformative_frac, 4),
            "kl_mean": round(mean_kl, 5),
            "policy_loss": round(p_loss_val, 4),
            "entropy": round(mean_ent, 4),
            "clip_fraction": round(clip_frac, 4),
            "response_length": round(mean_len, 1),
            "grad_norm": round(gnorm_val, 4),
        }
        history_records.append(record)
        append_jsonl(history_file, record)

        print(
            f"Update {step:>2}/{num_updates} | "
            f"Reward: {mean_rew:>6.3f} | "
            f"GroupStd: {mean_group_std:>5.3f} | "
            f"UninfFrac: {uninformative_frac:>4.2f} | "
            f"KL: {mean_kl:>7.4f} | "
            f"Loss: {p_loss_val:>6.3f} | "
            f"Len: {mean_len:>5.1f} | "
            f"Grad: {gnorm_val:>5.3f}"
        )

    # Save policy adapter
    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)
    elapsed = timer()
    print(f"\nSaved trained GRPO adapter to: {out} (Elapsed: {elapsed:.1f}s)")

    # Record peak VRAM if CUDA available
    peak_vram_gb = None
    if torch.cuda.is_available():
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
        print(f"Peak VRAM: {peak_vram_gb} GB")

    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "updates": num_updates,
        "final_reward": history_records[-1]["reward_mean"] if history_records else None,
        "mean_reward": float(np.mean([r["reward_mean"] for r in history_records])) if history_records else None,
        "mean_group_std": float(np.mean([r["reward_group_std"] for r in history_records])) if history_records else None,
        "mean_uninformative_fraction": float(np.mean([r["uninformative_fraction"] for r in history_records])) if history_records else None,
        "mean_kl": float(np.mean([r["kl_mean"] for r in history_records])) if history_records else None,
        "mean_entropy": float(np.mean([r["entropy"] for r in history_records])) if history_records else None,
        "mean_response_length": float(np.mean([r["response_length"] for r in history_records])) if history_records else None,
        "max_grad_norm": float(np.max([r["grad_norm"] for r in history_records])) if history_records else None,
        "wall_clock_time_seconds": round(elapsed, 2),
        "peak_vram_gb": peak_vram_gb,
        "adapter_path": str(out),
    }

    summary_file = results_dir / f"{run_name}_train_summary.json"
    save_json(summary_file, summary)
    print(f"Saved training summary to: {summary_file}")
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
