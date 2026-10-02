from __future__ import annotations

import argparse
from pathlib import Path

from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    import random
    import torch
    from common.logging_utils import append_jsonl, wall_timer
    from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
    from common.metrics import sample_entropy, masked_mean
    from common.models import reference_mode, clear_gpu, trainable_parameters, token_values
    from task2_ppo.ppo import compute_gae, shaped_rewards, ppo_policy_loss, value_mse_loss, normalize_advantages
    
    updates = cfg.get("updates", 20)
    clip_eps = cfg.get("clip_epsilon", 0.2)
    kl_b = cfg.get("kl_beta", 0.1)
    batch_size = cfg.get("prompts_per_update", 1)
    ppo_epochs = cfg.get("ppo_epochs", 2)
    max_prompt_length = cfg.get("max_prompt_length", 256)
    max_response_length = cfg.get("max_response_length", 512)
    gamma = cfg.get("gamma", 1.0)
    lam = cfg.get("gae_lambda", 0.95)
    max_grad = cfg.get("max_grad_norm", 1.0)

    value_coef = cfg.get("value_coef", 0.5)

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    tokenizer = bundle["tokenizer"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompts = bundle["prompt_rows"]
    opt_p = bundle["policy_optimizer"]
    opt_v = bundle["value_optimizer"]
    
    device = next(policy.parameters()).device
    timer = wall_timer()
    history_file = out.parent / f"{run_name}_train_history.jsonl"
    if history_file.exists(): history_file.unlink()

    for step in range(updates):
        policy.eval()
        value_model.eval()
        
        batch_prompts = random.choices(prompts, k=batch_size)
        pm_list = [prompt_messages(r) for r in batch_prompts]

        with torch.no_grad():
            gen_out = batch_generate(policy, tokenizer, pm_list, max_prompt_length, max_response_length)
            seqs = gen_out["sequences"].clone()
            attn = gen_out["attention_mask"].clone()
            pw = gen_out["prompt_width"]
            rids = gen_out["response_ids"].clone()
            rmask = gen_out["response_mask"].clone()

            old_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)

            v_all = token_values(value_model, seqs, attn)
            v_old = v_all[:, pw-1:-1][:, :rids.shape[1]].float()

            rm_scores = score_reward_pairs(reward_model, reward_tokenizer, pm_list, gen_out["responses"])
            
            rewards = shaped_rewards(rm_scores, old_logp, ref_logp, rmask, kl_b)
            adv, returns = compute_gae(rewards, v_old, rmask, gamma, lam)
            norm_adv = normalize_advantages(adv, rmask)

        policy.train()
        value_model.train()

        for _ in range(ppo_epochs):
            new_logp, _ = response_token_logprobs(policy, seqs, attn, pw, rids)
            p_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, norm_adv, rmask, clip_eps)

            opt_p.zero_grad()
            p_loss.backward()
            p_grad = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad)
            if not torch.isnan(p_grad) and not torch.isinf(p_grad):
                opt_p.step()
            else:
                print("WARNING: NaN gradient in policy! Skipping step.")
                opt_p.zero_grad()

            v_all_new = token_values(value_model, seqs, attn)
            v_new = v_all_new[:, pw-1:-1][:, :rids.shape[1]].float()
            v_loss = value_coef * value_mse_loss(v_new, returns.float(), rmask)
            
            opt_v.zero_grad()
            v_loss.backward()
            v_grad = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad)
            if not torch.isnan(v_grad) and not torch.isinf(v_grad):
                opt_v.step()
            else:
                print("WARNING: NaN gradient in value model! Skipping step.")
                opt_v.zero_grad()

        kl_mean = masked_mean(old_logp - ref_logp, rmask).item()
        record = {
            "step": step + 1,
            "reward_mean": rm_scores.mean().item(),
            "kl_mean": kl_mean,
            "policy_loss": p_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": sample_entropy(new_logp.detach(), rmask).item(),
            "clip_fraction": clip_frac.item(),
            "response_length": rmask.sum(-1).mean().item(),
            "grad_norm": float(p_grad),
            "wall_time_sec": timer(),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        }
        append_jsonl(history_file, record)
        print(f"Update {step+1}/{updates} | Reward: {record['reward_mean']:.3f} | KL: {record['kl_mean']:.3f} | Len: {record['response_length']:.1f}")

    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    value_model.save_pretrained(str(out / "value_model"))
    print(f"PPO run '{run_name}' complete.")
    clear_gpu(policy, value_model, reward_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
