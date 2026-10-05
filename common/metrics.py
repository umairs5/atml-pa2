from __future__ import annotations

import re
import numpy as np
import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def sampled_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor):
    return masked_mean(policy_logp - ref_logp, mask)


def sample_entropy(sampled_logp: torch.Tensor, mask: torch.Tensor):
    """Mean surprisal of sampled tokens (not full policy entropy)."""
    return -masked_mean(sampled_logp, mask)


def token_entropy(logits: torch.Tensor, mask: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    """Mean categorical entropy over valid response-token distributions.

    This is the entropy required by the assignment, unlike ``sample_entropy``,
    which is only the negative log-probability of the sampled response tokens.
    """
    # A [batch, tokens, vocabulary] float32 log-softmax can be several GiB for
    # this policy.  Work over valid positions in small chunks so entropy logging
    # does not substantially increase the continuation's peak VRAM.
    flat_logits = logits.reshape(-1, logits.shape[-1])
    valid_indices = mask.reshape(-1).bool().nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return logits.new_zeros(())

    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, valid_indices.numel(), chunk_size):
        indices = valid_indices[start : start + chunk_size]
        scores = flat_logits.index_select(0, indices).float()
        log_z = torch.logsumexp(scores, dim=-1)
        probs = torch.exp(scores - log_z[:, None])
        total = total + (log_z - (probs * scores).sum(dim=-1)).sum()
    return total / valid_indices.numel()


def mean_response_length(mask: torch.Tensor):
    return float(mask.sum(-1).float().mean().item())


def preference_accuracy(chosen_logp, rejected_logp):
    return float((chosen_logp > rejected_logp).float().mean().item())


def dpo_preference_margin(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp):
    """m_theta = [log pi_theta(y+|x) - log pi_ref(y+|x)] - [log pi_theta(y-|x) - log pi_ref(y-|x)]"""
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    return policy_margin - ref_margin


def dpo_preference_accuracy(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp):
    """Fraction of pairs where m_theta > 0."""
    margin = dpo_preference_margin(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp)
    return float((margin > 0).float().mean().item())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def parse_word_limit(prompt: str):
    patterns = [
        r"(?:at most|no more than|under|within)\s+(\d+)\s+words?",
        r"(?:in|use)\s+(\d+)\s+words?\s+(?:or fewer|max(?:imum)?)",
        r"(?:maximum|max)\s+(?:of\s+)?(\d+)\s+words?",
    ]
    lower = str(prompt).lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return None


def word_limit_compliance(prompt: str, response: str):
    limit = parse_word_limit(prompt)
    if limit is None:
        return None
    return float(word_count(response) <= limit)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
