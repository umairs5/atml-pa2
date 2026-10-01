from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Corrected implementation based on the DPO objective:
        L_DPO(theta) = -E[log sigma(beta * ((log pi_theta(y+|x) - log pi_ref(y+|x))
                                            - (log pi_theta(y-|x) - log pi_ref(y-|x))))]
                     = -E[log sigma(beta * (policy_margin - ref_margin))]
    """
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    implicit_margin = policy_margin - ref_margin

    logits = beta * implicit_margin

    loss = -F.logsigmoid(logits).mean()
    return loss, {
        "loss": loss.detach(),
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "ref_margin_mean": ref_margin.detach().mean(),
        "implicit_margin_mean": implicit_margin.detach().mean(),
        "preference_accuracy": (implicit_margin > 0).float().mean().detach(),
    }
