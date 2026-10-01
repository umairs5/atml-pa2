from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    beta_val = bundle["beta"]

    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    epochs = int(cfg.get("epochs", 1))

    history_file = results_dir / f"{run_name}_train_history.jsonl"
    if history_file.exists():
        history_file.unlink()

    step = 0
    timer = wall_timer()
    optimizer.zero_grad()

    accumulated_loss = 0.0
    accumulated_diag = {}

    from tqdm import tqdm
    from common.generation import response_sequence_logprobs
    from common.models import reference_mode, clear_gpu
    from common.logging_utils import append_jsonl, save_json

    print(f"Starting DPO training: run='{run_name}', examples={len(bundle['rows'])}, beta={beta_val}, epochs={epochs}")
    model.train()

    for epoch in range(epochs):
        for batch_idx, (chosen_batch, rejected_batch) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            chosen_batch = {k: v.to(device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(device) for k, v in rejected_batch.items()}

            # 1. Reference log-probabilities (frozen base policy with adapters disabled)
            with reference_mode(model):
                with torch.no_grad():
                    ref_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
                    ref_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            # 2. Trainable policy log-probabilities
            policy_chosen_logp, _, _ = response_sequence_logprobs(model, chosen_batch)
            policy_rejected_logp, _, _ = response_sequence_logprobs(model, rejected_batch)

            # 3. DPO Loss & metrics
            loss, diag = dpo_loss(
                policy_chosen_logp=policy_chosen_logp,
                policy_rejected_logp=policy_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=beta_val,
            )

            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            accumulated_loss += loss.item()
            for k, v in diag.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                accumulated_diag[k] = accumulated_diag.get(k, 0.0) + val

            # 4. Optimizer step on accumulation boundary
            is_accum_boundary = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader))
            if is_accum_boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                actual_accum = ((batch_idx % grad_accum_steps) + 1)
                record = {
                    "step": step,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": accumulated_loss / actual_accum,
                    "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    "wall_time_sec": timer(),
                }
                for k, v in accumulated_diag.items():
                    record[k] = v / actual_accum

                append_jsonl(history_file, record)

                accumulated_loss = 0.0
                accumulated_diag = {}
                step += 1

    # Save adapter & tokenizer
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    summary = {
        "run_name": run_name,
        "config_path": str(config_path),
        "beta": beta_val,
        "epochs": epochs,
        "total_updates": step,
        "total_wall_time_sec": timer(),
        "adapter_path": str(output),
    }
    save_json(results_dir / f"{run_name}_train_summary.json", summary)
    print(f"Training completed. Model saved to {output}. Logs saved to {results_dir}")

    clear_gpu(model, optimizer)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
