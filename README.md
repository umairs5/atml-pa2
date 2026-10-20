# ATML PA2 - LLM Post-Training

My code and results for Programming Assignment 2 of EE-5102/CS-6304 (Advanced Topics in Machine Learning, Fall 2026).

The assignment compares DPO, PPO and GRPO on Qwen2.5-1.5B-Instruct (LoRA), evaluates the resulting policies on XSTest, and compares a verifiable math reward (RLVR) with AI feedback (RLAIF). Every number in the report comes from a file in `results/`.

## Setup

```bash
git clone https://github.com/umairs5/atml-pa2.git
cd atml-pa2
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
```

`download_assets` pulls the fixed course data, cached rollouts and supplied checkpoints from the course Hugging Face repo (`AbDu11aHHH/ATML-PA2-assets`, pinned revision in `configs/assets.yaml`). Public models (policy, reward model, judge) are downloaded from Hugging Face at runtime. None of these are committed here.

All runs were done on a Google Colab GPU with seed 6304, using the pinned versions in `requirements.txt` (transformers 4.57.1, trl 0.27.2, peft 0.17.1). All hyperparameters are in `configs/`.

## Fixes to the starter objectives

Each core objective in the starter code had one bug, fixed before any run:

- **DPO** (`task1_dpo/dpo.py`): the reference margin was added instead of subtracted. After the fix the first training loss is log 2 = 0.6931, as expected when policy = reference.
- **PPO** (`task2_ppo/ppo.py`): the clipped surrogate used `torch.maximum` instead of `torch.minimum`, and the advantage/returns were not detached.
- **GRPO** (`task3_grpo/grpo.py`): advantages were normalized over the whole batch, ignoring `group_ids`. They are now normalized inside each prompt group.

## Commands for each task

Run everything from the repo root. Training histories are written to `outputs/` (ignored by git); copies of the histories used in the report are in `results/`.

### Task 1 - DPO

```bash
# standard run (1 epoch, beta = 0.10) and held-out evaluation
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard

# beta study (0.03, 0.10, 0.30 on the 600-pair short budget)
python -m task1_dpo.ablate_beta --config configs/dpo.yaml

# length study (trains the length-balanced model, stratified eval, word-limit prompts)
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
# standard 20-update continuation from the supplied midpoint
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard

# clipping study: cached-rollout analysis + 8-update forks for eps = 0.05, 0.20, 0.50
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml

# KL study: 8-update forks for beta_KL = 0.0, 0.10, 0.20
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
# standard 20-update continuation (K = 4)
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard

# group size study on the supplied K = 8 cache (no training)
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

# canonical (1/T_k) vs Dr. GRPO (1/L_max), 8-update forks
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
# label results/task4_safety/manual_audit_template.csv by hand (blind to the judge),
# save it as results/task4_safety/manual_audit_completed.csv
# (task4_safety/manual_audit_tool.html is the small page I used for labelling)
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
python scripts/plot_confusion.py
```

### Task 5 - RLVR vs RLAIF

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

The SVAMP transfer file (`data/math_transfer_eval.jsonl`) is created by `scripts/prepare_transfer_eval.py` during `download_assets` (first 100 problems of the official challenge set).

### Extra numbers used in the report

A few numbers in the report are not printed by the task scripts (for example the reward vs length correlation, the PPO clipped surrogate on the cached batch, the paired GRPO gradient norms, the verifier/judge disagreement breakdown and the safety label counts per prompt type). They are computed from the saved results only, with no model loading:

```bash
python -m scripts.report_extra_stats
```

Output: `results/report_extra_stats.json`.

## Where each result in the report comes from

| Report | Files |
|---|---|
| Table 1 (DPO standard + beta forks) | `results/task1_dpo/standard_eval_summary.json`, `beta_ablation_summary.json`, `*_train_summary.json` |
| Table 2 (length study) | `results/task1_dpo/length_confounding_summary.json`, `*_word_limit_generations.jsonl` |
| Table 3 (PPO / GRPO standard runs) | `results/task2_ppo/standard_*`, `results/task3_grpo/standard_*`, `results/report_extra_stats.json` |
| Table 4 (PPO clipping and KL forks) | `results/task2_ppo/cached_clipping_analysis.json`, `clipping_ablation_summary.json`, `kl_ablation_summary.json` |
| Table 5 (GRPO group size) | `results/task3_grpo/group_size_analysis.json` |
| Table 6 (GRPO normalization) | `results/task3_grpo/normalization_comparison_summary.json`, `norm_*_train_history.jsonl` |
| Tables 7-9 (safety, categories, manual audit) | `results/task4_safety/safety_policy_summary.csv`, `safety_category_summary.csv`, `manual_audit_*.csv`, `manual_audit_agreement.json` |
| Tables 10-11 (RLVR vs RLAIF) | `results/task5_feedback/feedback_policy_comparison.csv`, `gsm_summary.json`, `transfer_summary.json`, `diagnostic_summary.json` |
| Figures | `report/figures/` |

Generated responses for every evaluation are saved next to the summaries (`*_generations.jsonl`, `generated_*.jsonl`, `judged_*.jsonl`), so the examples quoted in the report can be checked.

## Repository layout

```
configs/         hyperparameters for each task
common/          shared data, model, generation and metric helpers
scripts/         asset download/validation, transfer set, extra report stats, plots
task1_dpo/ ... task5_feedback/   one folder per task
results/         JSON/CSV summaries, histories and generations used in the report
report/figures/  figures in the report
data/, cached/, checkpoints/, manifests/   course assets (downloaded, not committed)
outputs/         trained adapters (not committed)
```

## Attribution

- The starter code (data/model loading, objective scaffolds, judge and verifier utilities, asset scripts) comes from the course release: https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining. I fixed the three objective bugs above and wrote the training loops, ablation code and evaluation/analysis for all five tasks.
- Course data, cached rollouts and checkpoints: https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets
- Public models and data: Qwen2.5 models, `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, UltraFeedback, GSM8K, SVAMP and XSTest (https://github.com/paul-rottger/xstest).
- Libraries: PyTorch, Hugging Face Transformers, TRL, PEFT, bitsandbytes, pandas, matplotlib.
