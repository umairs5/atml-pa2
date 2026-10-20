"""Extra statistics quoted in the report that are not printed by the task scripts.

Reads only files that already exist under data/ and results/ and writes
results/report_extra_stats.json. No model is loaded and nothing is retrained.

    python -m scripts.report_extra_stats
"""
from __future__ import annotations

import ast
import csv
import json
import statistics as st
from collections import Counter

from common.data import repo_path


def read_jsonl(path):
    with open(repo_path(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    return st.correlation(rank(a), rank(b))


def last_assistant_words(raw):
    msgs = ast.literal_eval(raw) if isinstance(raw, str) else raw
    return len(str(msgs[-1]["content"]).split())


def dpo_dataset_lengths():
    out = {}
    for name, path in [("standard_train", "data/dpo_standard_train.jsonl"),
                       ("length_balanced_train", "data/dpo_length_balanced_train.jsonl")]:
        rows = read_jsonl(path)
        cw = [last_assistant_words(r["chosen"]) for r in rows]
        rw = [last_assistant_words(r["rejected"]) for r in rows]
        diff = [c - r for c, r in zip(cw, rw)]
        out[name] = {
            "pairs": len(rows),
            "chosen_longer_frac_words": sum(d > 0 for d in diff) / len(diff),
            "rejected_longer_frac_words": sum(d < 0 for d in diff) / len(diff),
            "mean_chosen_words": st.mean(cw),
            "mean_rejected_words": st.mean(rw),
            "median_word_diff_chosen_minus_rejected": st.median(diff),
        }
        if "length_stratum" in rows[0]:
            out[name]["strata"] = dict(Counter(r["length_stratum"] for r in rows))
    return out


def reward_length(path, reward_key, cap):
    rows = read_jsonl(path)
    R = [float(r[reward_key]) for r in rows]
    L = [int(r["response_length"]) for r in rows]
    trunc = [r for r in rows if r.get("truncated") or int(r["response_length"]) >= cap]
    done = [r for r in rows if not (r.get("truncated") or int(r["response_length"]) >= cap)]
    return {
        "n": len(rows),
        "spearman_reward_length": spearman(R, L),
        "truncated_count": len(trunc),
        "mean_reward_truncated": st.mean(float(r[reward_key]) for r in trunc) if trunc else None,
        "mean_reward_not_truncated": st.mean(float(r[reward_key]) for r in done),
        "frac_reward_negative": sum(x < 0 for x in R) / len(R),
    }


def ppo_history():
    rows = read_jsonl("results/task2_ppo/standard_train_history.jsonl")
    L = [r["response_length"] for r in rows]
    short = [r for r in rows if r["response_length"] <= 15]
    rest = [r for r in rows if r["response_length"] > 15]
    return {
        "updates": len(rows),
        "prompts_per_update": 1,
        "length_min": min(L), "length_max": max(L),
        "updates_at_512_cap": sum(x >= 512 for x in L),
        "updates_at_most_15_tokens": len(short),
        "mean_grad_norm_short": st.mean(r["grad_norm"] for r in short),
        "mean_grad_norm_rest": st.mean(r["grad_norm"] for r in rest),
        "mean_value_loss_short": st.mean(r["value_loss"] for r in short),
        "mean_value_loss_rest": st.mean(r["value_loss"] for r in rest),
        "spearman_reward_vs_update_index": spearman([r["reward_mean"] for r in rows], list(range(len(rows)))),
        "reward_std_over_updates": st.pstdev(r["reward_mean"] for r in rows),
        "max_grad_norm_step": max(rows, key=lambda r: r["grad_norm"])["step"],
    }


def ppo_cached_surrogate(kl_beta=0.10, gamma=1.0, lam=0.95):
    """Clipped surrogate on the cached batch, using the same ratio as task2_ppo/analyze_clipping.py.

    The cache stores old-policy and reference log-probs, values and the terminal reward, but no
    newer policy, so rho = pi_old / pi_ref. Advantages are rebuilt with the course GAE helper.
    """
    import torch
    from task2_ppo.ppo import compute_gae, normalize_advantages, shaped_rewards

    rows = torch.load(repo_path("cached/ppo_rollout.pt"), map_location="cpu", weights_only=False)
    ratios, advs = [], []
    for r in rows:
        old = torch.as_tensor(r["old_logprobs"]).float()[None]
        ref = torch.as_tensor(r["ref_logprobs"]).float()[None]
        val = torch.as_tensor(r["values"]).float()[None]
        mask = torch.ones_like(old)
        rew = shaped_rewards(torch.tensor([float(r["effective_terminal_reward"])]), old, ref, mask, kl_beta)
        adv, _ = compute_gae(rew, val, mask, gamma=gamma, lam=lam)
        ratios.append(torch.exp(old - ref)[0])
        advs.append(adv[0])
    rho = torch.cat(ratios)
    adv = normalize_advantages(torch.cat(advs)[None], torch.ones(1, rho.numel()))[0]
    out = {"tokens": int(rho.numel()), "unclipped_surrogate": float((rho * adv).mean())}
    for eps in (0.05, 0.20, 0.50):
        s1, s2 = rho * adv, rho.clamp(1 - eps, 1 + eps) * adv
        obj = torch.minimum(s1, s2)
        out[f"eps_{eps}"] = {
            "outside_range_frac": float(((rho < 1 - eps) | (rho > 1 + eps)).float().mean()),
            "affected_token_frac": float((s2 < s1).float().mean()),
            "clipped_surrogate": float(obj.mean()),
        }
    return out


def trajectory_summary(path, keys):
    rows = read_jsonl(path)
    out = {}
    for k in keys:
        v = [float(r[k]) for r in rows if r.get(k) is not None]
        out[k] = {"first": v[0], "last": v[-1], "mean": st.mean(v), "min": min(v), "max": max(v)}
    return out


def grpo_history():
    std = read_jsonl("results/task3_grpo/standard_train_history.jsonl")
    out = {
        "updates": len(std),
        "prompts_per_update": 1,
        "zero_std_updates": [r["step"] for r in std if r["uninformative_fraction"] >= 1.0],
        "all_completions_masked_updates": [r["step"] for r in std if r["response_length"] == 0],
        "max_grad_norm_step": max(std, key=lambda r: r["grad_norm"])["step"],
    }
    can = {r["step"]: r for r in read_jsonl("results/task3_grpo/norm_canonical_train_history.jsonl")}
    dr = {r["step"]: r for r in read_jsonl("results/task3_grpo/norm_dr_grpo_train_history.jsonl")}
    # Updates 1, 3, 4, 6 and 8 sample the same completions in both forks, so they are a paired check.
    out["paired_grad_norm"] = {
        s: {"mean_len": can[s]["response_length"], "canonical": can[s]["grad_norm"], "dr_grpo": dr[s]["grad_norm"]}
        for s in can if s in dr and can[s]["response_length"] == dr[s]["response_length"] and can[s]["reward_mean"] == dr[s]["reward_mean"]
    }
    out["canonical_short_to_long_per_update"] = {s: can[s]["short_to_long_token_weight"] for s in can if can[s].get("short_to_long_token_weight")}
    return out


def safety():
    pols = ["sft", "dpo", "ppo", "grpo"]
    judged = {p: read_jsonl(f"results/task4_safety/judged_{p}.jsonl") for p in pols}
    out = {"judge_label_by_class": {}, "identical_to_sft": {}}
    sft = {r["xstest_id"]: r for r in judged["sft"]}
    for p in pols:
        c = Counter(f'{r["benchmark_class"]}:{r["judge_label"]}' for r in judged[p])
        out["judge_label_by_class"][p] = dict(sorted(c.items()))
        if p != "sft":
            out["identical_to_sft"][p] = sum(r["response"] == sft[r["xstest_id"]]["response"] for r in judged[p])
    out["judge_rationale_tags_safe_JR_sft"] = dict(Counter(
        r.get("judge_rationale_tag") for r in judged["sft"]
        if r["benchmark_class"] == "SAFE" and r["judge_label"] == "JUSTIFIED_REFUSAL").most_common(6))

    key = {r["audit_row_id"]: r for r in csv.DictReader(open(repo_path("results/task4_safety/manual_audit_key.csv"), encoding="utf-8"))}
    lookup = {(p, str(r["xstest_id"])): r for p in pols for r in judged[p]}
    audit = {}
    types = Counter()
    for row in csv.DictReader(open(repo_path("results/task4_safety/manual_audit_completed.csv"), encoding="utf-8")):
        k = key[row["audit_row_id"]]
        j = lookup[(k["policy"], k["xstest_id"])]["judge_label"]
        d = audit.setdefault(k["policy"], {"SAFE": Counter(), "UNSAFE": Counter()})
        d[row["benchmark_class"]][f'manual={row["manual_label"]}'] += 1
        d[row["benchmark_class"]]["agree"] += int(row["manual_label"] == j)
        d[row["benchmark_class"]]["n"] += 1
        if row["manual_label"] == "OVER_REFUSAL":
            types[row["type"]] += 1
    out["manual_audit_by_policy_class"] = {p: {c: dict(v) for c, v in d.items()} for p, d in audit.items()}
    out["manual_over_refusal_types"] = dict(types)
    return out


def feedback():
    out = {}
    for ds in ["gsm", "transfer"]:
        pairs = read_jsonl(f"results/task5_feedback/{ds}_pairwise.jsonl")
        gens = read_jsonl(f"results/task5_feedback/{ds}_generations.jsonl")
        d = {"verifier_vs_judge": {}, "format": {}}
        by_key = {(r["policy"], r["prompt_id"]): r for r in gens}
        d["disagreement_due_to_format"] = {}
        for p in ["rlvr", "rlaif"]:
            rows = [r for r in pairs if r["policy"] == p]
            c = Counter(f'verifier={r["verifier_preference"]}|judge={r["ai_preference"]}' for r in rows)
            d["verifier_vs_judge"][p] = dict(sorted(c.items()))
            # A = candidate policy, B = SFT baseline.
            decided = [r for r in rows if r["verifier_preference"] in ("A", "B")
                       and r["ai_preference"] != r["verifier_preference"]]
            loser_unparsed = sum(
                not by_key[("sft" if r["verifier_preference"] == "A" else p, r["prompt_id"])]["format_compliant"]
                for r in decided)
            d["disagreement_due_to_format"][p] = {
                "verifier_decided_judge_disagrees": len(decided),
                "verifier_loser_had_no_parsable_answer": loser_unparsed,
            }
        for p in ["sft", "rlvr", "rlaif"]:
            g = [r for r in gens if r["policy"] == p]
            fmt = [r for r in g if r["format_compliant"]]
            d["format"][p] = {
                "n": len(g),
                "format_ok": len(fmt),
                "accuracy_given_format": sum(r["exact_reward"] for r in fmt) / len(fmt),
                "format_fail_truncated": sum((not r["format_compliant"]) and r["truncated"] for r in g),
                "format_fail_finished": sum((not r["format_compliant"]) and not r["truncated"] for r in g),
            }
        out[ds] = d
    return out


def main():
    stats = {
        "task1_dpo_dataset_lengths": dpo_dataset_lengths(),
        "reward_vs_length_heldout": {
            "dpo_standard": reward_length("results/task1_dpo/standard_eval_generations.jsonl", "reward_score", 256),
            "ppo_standard": reward_length("results/task2_ppo/standard_eval_generations.jsonl", "reward", 768),
            "grpo_standard": reward_length("results/task3_grpo/standard_eval_generations.jsonl", "reward", 768),
        },
        "task2_ppo_history": ppo_history(),
        "task2_ppo_cached_surrogate": ppo_cached_surrogate(),
        "task2_ppo_trajectory": trajectory_summary(
            "results/task2_ppo/standard_train_history.jsonl",
            ["reward_mean", "kl_mean", "policy_loss", "value_loss", "entropy", "clip_fraction", "grad_norm", "response_length"]),
        "task3_grpo_trajectory": trajectory_summary(
            "results/task3_grpo/standard_train_history.jsonl",
            ["reward_mean", "kl_mean", "reward_group_std", "uninformative_fraction", "policy_loss", "grad_norm", "entropy", "response_length"]),
        "task3_grpo_history": grpo_history(),
        "task4_safety": safety(),
        "task5_feedback": feedback(),
    }
    out = repo_path("results/report_extra_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
