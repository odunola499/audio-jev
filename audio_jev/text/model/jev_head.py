# Adapted from Cloudflare Clef under the Apache-2.0 license.
# Source: https://huggingface.co/Cloudflare/clef/blob/main/joint_schema_model.py

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}


@dataclass(frozen=True)
class EncodedQuestion:
    """Use token spans with an inclusive start and an exclusive end."""

    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]
    record_id: str = ""


class EvidenceRoutingLayer(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        feedforward: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)

        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.feedforward_norm = nn.LayerNorm(width)

        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)

        routed, _ = self.attention(
            normalized_queries,
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )

        queries = queries + self.attention_dropout(routed)

        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(nn.Module):
    """Return option logits for each question in each record."""

    def __init__(
        self,
        hidden_size: int,
        width: int = 256,
        routing_layers: int = 2,
        layers: int = 4,
        heads: int = 4,
        feedforward: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_norm = nn.LayerNorm(hidden_size)

        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)

        self.type_embedding = nn.Embedding(3, width)

        self.evidence_layers = nn.ModuleList(
            [
                EvidenceRoutingLayer(
                    width=width,
                    heads=heads,
                    feedforward=feedforward,
                    dropout=dropout,
                )
                for _ in range(routing_layers)
            ]
        )

        self.option_summary_norm = nn.LayerNorm(width)

        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=width,
                    nhead=heads,
                    dim_feedforward=feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )

        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)

        self.residual_scorer = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _mean_span(values: torch.Tensor, span: tuple[int, int]) -> torch.Tensor:
        start, end = span

        return values[start:end].mean(dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        records: list[EncodedRecord],
        output_embedding_weight: torch.Tensor,
    ) -> list[list[torch.Tensor]]:
        """Use right padding. Supply the full sequence and its token spans."""

        if len(records) != hidden_states.shape[0]:
            raise ValueError("Supply one record for each batch item.")

        lengths = attention_mask.sum(dim=-1)
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
        expected_mask = positions.unsqueeze(0) < lengths.unsqueeze(1)

        if not torch.equal(attention_mask, expected_mask) or (lengths == 0).any():
            raise ValueError("Use a nonempty sequence with right padding for each record.")

        results: list[list[torch.Tensor]] = []
        normalized_hidden = self.hidden_norm(hidden_states)

        for batch_index, record in enumerate(records):
            sequence_length = int(attention_mask[batch_index].sum().item())
            sequence_hidden = normalized_hidden[batch_index, :sequence_length]

            memory = self.memory_projection(sequence_hidden).unsqueeze(0)
            global_vector = sequence_hidden[-1]

            question_vectors = torch.stack(
                [
                    self._mean_span(sequence_hidden, question.question_span)
                    for question in record.questions
                ]
            )

            type_ids = torch.tensor(
                [question.question_type for question in record.questions],
                device=hidden_states.device,
            )

            option_contexts: list[torch.Tensor] = []
            lexical_options: list[torch.Tensor] = []
            option_counts = []

            for question in record.questions:
                context_vectors = torch.stack(
                    [
                        self._mean_span(sequence_hidden, span)
                        for span in question.option_spans
                    ]
                )

                lexical_vectors = []

                for start, end in question.option_spans:
                    token_ids = input_ids[batch_index, start:end]
                    lexical_vectors.append(output_embedding_weight[token_ids].mean(dim=0))

                lexical = torch.stack(lexical_vectors)
                option_contexts.append(context_vectors)
                lexical_options.append(lexical)
                option_counts.append(len(question.option_spans))

            option_queries = []
            for question_index, (context_vectors, lexical) in enumerate(
                zip(option_contexts, lexical_options)
            ):
                option_queries.append(
                    self.option_context_projection(context_vectors)
                    + self.option_lexical_projection(lexical)
                    + self.option_question_projection(
                        question_vectors[question_index]
                    ).unsqueeze(0)
                )

            routed_options = torch.cat(option_queries, dim=0).unsqueeze(0)

            for layer in self.evidence_layers:
                routed_options = layer(routed_options, memory)

            routed_options = routed_options[0]
            split_options = list(torch.split(routed_options, option_counts, dim=0))

            base_fields = self.question_projection(question_vectors)
            option_summaries = []
            for field, options in zip(base_fields, split_options):
                routing_weights = torch.softmax(
                    torch.matmul(options, field) / math.sqrt(options.shape[-1]),
                    dim=0,
                )
                option_summaries.append(
                    torch.sum(routing_weights.unsqueeze(-1) * options, dim=0)
                )

            fields = (
                base_fields
                + self.option_summary_norm(torch.stack(option_summaries))
                + self.global_projection(global_vector).unsqueeze(0)
                + self.type_embedding(type_ids)
            )

            fields = fields.unsqueeze(0)

            for layer in self.layers:
                fields = layer(fields, memory)

            fields = self.field_norm(fields[0])

            record_logits: list[torch.Tensor] = []
            for field, question, lexical, routed in zip(
                fields,
                record.questions,
                lexical_options,
                split_options,
            ):
                anchor = F.normalize(
                    question_vectors[len(record_logits)] + global_vector,
                    dim=-1,
                )
                lexical_anchor = F.normalize(lexical, dim=-1)
                prior_scale = self.prior_logit_scale.clamp(max=math.log(100.0)).exp()
                prior = prior_scale * torch.matmul(lexical_anchor, anchor)

                options = self.option_norm(routed)
                repeated_field = field.unsqueeze(0).expand_as(options)
                cosine = F.cosine_similarity(repeated_field, options, dim=-1)

                features = torch.cat(
                    [
                        repeated_field,
                        options,
                        repeated_field * options,
                        torch.abs(repeated_field - options),
                    ],
                    dim=-1,
                )

                residual = self.residual_scorer(features).squeeze(-1)
                joint_scale = self.joint_logit_scale.clamp(max=math.log(100.0)).exp()
                joint = joint_scale * cosine + residual

                record_logits.append(
                    prior + torch.sigmoid(self.residual_gate) * joint
                )

            results.append(record_logits)

        return results
