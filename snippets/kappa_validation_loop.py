"""
kappa_validation_loop.py — illustrative snippet of the Kappa-driven prompt
refinement loop described in README §5. Not runnable against client data; the
inputs below are synthetic and the point of the file is to show the control
flow of the loop, not to serve as a drop-in utility.

The loop is the operational definition of "ready for production" used in this
case study. The LLM extracts four categorical attributes per campaign (tone,
theme, CTA strength, personalisation). A human expert independently labels a
validation sample using the same criteria. Cohen's Kappa is computed per
dimension. When a dimension falls below the threshold, the divergence cases
on that dimension are surfaced as candidate few-shot examples for the next
prompt iteration.
"""

from dataclasses import dataclass
from typing import Iterable

from sklearn.metrics import cohen_kappa_score


DIMENSIONS = ("tone", "theme", "cta_strength", "personalisation")
KAPPA_THRESHOLD = 0.75


@dataclass
class CampaignLabel:
    """One campaign with its four categorical labels from either source."""

    campaign_id: str
    tone: str
    theme: str
    cta_strength: str
    personalisation: str


def kappa_per_dimension(
    human_labels: list[CampaignLabel],
    llm_labels: list[CampaignLabel],
) -> dict[str, float]:
    """
    Compute Cohen's Kappa per attribute dimension. Both lists must be aligned
    by campaign_id. Returns one Kappa per dimension in DIMENSIONS.
    """
    assert len(human_labels) == len(llm_labels)
    assert all(h.campaign_id == l.campaign_id for h, l in zip(human_labels, llm_labels))

    return {
        dim: cohen_kappa_score(
            [getattr(h, dim) for h in human_labels],
            [getattr(l, dim) for l in llm_labels],
        )
        for dim in DIMENSIONS
    }


def divergence_cases(
    human_labels: list[CampaignLabel],
    llm_labels: list[CampaignLabel],
    dimension: str,
) -> list[tuple[str, str, str]]:
    """
    Return (campaign_id, human_value, llm_value) for every row where human
    and LLM disagree on the given dimension. These are the candidate
    few-shot examples for the next prompt iteration.
    """
    out: list[tuple[str, str, str]] = []
    for h, l in zip(human_labels, llm_labels):
        h_val = getattr(h, dimension)
        l_val = getattr(l, dimension)
        if h_val != l_val:
            out.append((h.campaign_id, h_val, l_val))
    return out


def dimensions_below_threshold(
    kappas: dict[str, float],
    threshold: float = KAPPA_THRESHOLD,
) -> list[str]:
    """Return the dimensions where Kappa is below the production threshold."""
    return [dim for dim, k in kappas.items() if k < threshold]


def iterate(
    human_labels: list[CampaignLabel],
    llm_label_fn,
) -> Iterable[dict[str, float]]:
    """
    Illustrative outer loop. On each iteration, run the current LLM prompt
    over the validation sample, score it, and (if any dimension is below
    threshold) surface divergence cases for prompt refinement. The caller
    is expected to update the prompt (adding few-shot examples derived from
    the divergence set) and call iterate() again. The loop stops when every
    dimension is at or above threshold.
    """
    while True:
        llm_labels = [llm_label_fn(h.campaign_id) for h in human_labels]
        kappas = kappa_per_dimension(human_labels, llm_labels)
        yield kappas

        below = dimensions_below_threshold(kappas)
        if not below:
            return

        for dim in below:
            cases = divergence_cases(human_labels, llm_labels, dim)
            # In production, these feed a human-in-the-loop review that
            # decides which cases to lift into the next version of the
            # system prompt as few-shot examples. The decision is not
            # automatic because some divergences are human labelling
            # errors rather than LLM errors.
            _ = cases
