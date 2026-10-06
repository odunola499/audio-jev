import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


type QuestionTarget = int | torch.Tensor | None


@dataclass(frozen=True)
class SchemaLossOutput:
    loss: torch.Tensor
    cross_entropy: torch.Tensor
    brier: torch.Tensor
    question_count: int


class SchemaLoss(nn.Module):
    """Compute cross-entropy and Brier loss for the schema head."""

    def __init__(
        self,
        label_smoothing: float = 0.1,
        brier_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be between zero and one.")

        if not math.isfinite(brier_weight) or brier_weight < 0.0:
            raise ValueError("brier_weight must be finite and nonnegative.")

        self.label_smoothing = label_smoothing
        self.brier_weight = brier_weight

    @staticmethod
    def _distribution(
        target: int | torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(target, bool):
            raise ValueError("Use an option index, not a Boolean target.")

        target = torch.as_tensor(target, device=logits.device).detach()

        if target.ndim == 0 and not target.is_floating_point():
            if target.dtype == torch.bool or target.is_complex():
                raise ValueError("The option index must be an integer.")

            index = int(target.item())
            if not 0 <= index < logits.numel():
                raise ValueError("The option index is outside the answer set.")

            return F.one_hot(
                target.long(), num_classes=logits.numel(),
            ).to(logits.dtype)

        if target.shape != logits.shape or not target.is_floating_point():
            raise ValueError("Supply an integer index or one probability per option.")

        target = target.to(logits.dtype)

        if not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError("Target probabilities must be finite and nonnegative.")

        if not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-5):
            raise ValueError("Target probabilities must sum to one.")

        return target

    def forward(
        self,
        logits: list[list[torch.Tensor]],
        targets: list[list[QuestionTarget]],
    ) -> SchemaLossOutput:
        """Match targets to the record, question, and option order of the logits.

        Use None to skip a question. For noul, index 0 is true and index 1
        is false. Use a probability vector for a soft target.
        """

        if len(logits) != len(targets):
            raise ValueError("Supply one target list for each record.")

        cross_entropy_terms = []
        brier_terms = []

        for record_logits, record_targets in zip(logits, targets):
            if len(record_logits) != len(record_targets):
                raise ValueError("Supply one target for each question.")

            for question_logits, target in zip(record_logits, record_targets):
                if target is None:
                    continue

                if question_logits.ndim != 1 or question_logits.numel() == 0:
                    raise ValueError("Supply a nonempty logit vector for each question.")

                if not question_logits.is_floating_point():
                    raise ValueError("Logits must use a floating-point type.")

                values = question_logits.float()
                distribution = self._distribution(target, values)

                smoothed = (
                    (1.0 - self.label_smoothing) * distribution
                    + self.label_smoothing / values.numel()
                )

                log_probabilities = F.log_softmax(values, dim=-1)
                cross_entropy_terms.append(-(smoothed * log_probabilities).sum())

                probabilities = log_probabilities.exp()
                brier_terms.append((probabilities - distribution).square().sum())

        if not cross_entropy_terms:
            raise ValueError("Supply at least one labeled question.")

        cross_entropy = torch.stack(cross_entropy_terms).mean()
        brier = torch.stack(brier_terms).mean()

        return SchemaLossOutput(
            loss=cross_entropy + self.brier_weight * brier,
            cross_entropy=cross_entropy,
            brier=brier,
            question_count=len(cross_entropy_terms),
        )
