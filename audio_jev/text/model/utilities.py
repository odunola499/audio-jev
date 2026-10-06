"""Utilities for converting a Jev request into schema-head metadata."""

import json
from typing import Any


from .jev_head import EncodedQuestion, EncodedRecord, QUESTION_TYPES


SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _tokens(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    question_type = question["type"]

    if question_type == "noul":
        criteria = {
            "true": "The proposition is true or the answer is yes.",
            "false": "The proposition is false or the answer is no.",
        }
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]

    if question_type == "choice":
        return sorted((str(key), value) for key, value in question["criteria"].items())

    if question_type == "score":
        return [
            (str(index), value)
            for index, value in enumerate(question["criteria"])
        ]

    raise ValueError(f"Unsupported question type: {question_type!r}")


def encode_prompt(
    tokenizer,
    prompt: dict[str, Any],
    *,
    max_length: int | None = None,
) -> EncodedRecord:
    """Encode one Jev request and record all schema token spans."""

    questions = prompt.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("The prompt must contain at least one question.")

    schema_ids = _tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    encoded_questions: list[EncodedQuestion] = []

    for question_index, (question_id, question) in enumerate(questions.items()):
        question_type = question.get("type")
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"{question_id}: unsupported question type")

        schema_ids.extend(
            _tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\n"
                f"ID: {question_id}\nTYPE: {question_type}\nINSTRUCTION: ",
            )
        )

        question_start = len(schema_ids)
        instruction = question.get("instructions") or str(question_id)
        schema_ids.extend(_tokens(tokenizer, _render(instruction)))
        question_end = len(schema_ids)
        schema_ids.extend(_tokens(tokenizer, "\nALLOWED OPTIONS:\n"))

        option_spans: list[tuple[int, int]] = []
        option_ids: list[str] = []

        for option_index, (option_id, description) in enumerate(_options(question)):
            schema_ids.extend(_tokens(tokenizer, f"OPTION {option_index + 1}: "))
            option_start = len(schema_ids)

            semantics: dict[str, Any] = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description

            schema_ids.extend(_tokens(tokenizer, _render(semantics)))
            option_spans.append((option_start, len(schema_ids)))
            option_ids.append(option_id)
            schema_ids.extend(_tokens(tokenizer, "\n"))

        schema_ids.extend(_tokens(tokenizer, "END FIELD\n"))
        encoded_questions.append(
            EncodedQuestion(
                question_id=str(question_id),
                question_type=QUESTION_TYPES[question_type],
                question_span=(question_start, question_end),
                option_spans=tuple(option_spans),
                option_ids=tuple(option_ids),
            )
        )

    prefix_ids = _tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\nSTATE:\n",
    )
    suffix_ids = _tokens(
        tokenizer,
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n"
        "</think>\n\nJOINT SCHEMA DECISIONS:",
    )
    state_ids = _tokens(tokenizer, _render(prompt.get("state")))

    fixed_length = len(prefix_ids) + len(schema_ids) + len(suffix_ids)
    if max_length is not None and fixed_length + len(state_ids) > max_length:
        raise ValueError(
            f"Prompt has {fixed_length + len(state_ids)} tokens; "
            f"maximum is {max_length}."
        )

    schema_offset = len(prefix_ids) + len(state_ids)
    shifted_questions = tuple(
        EncodedQuestion(
            question_id=question.question_id,
            question_type=question.question_type,
            question_span=tuple(index + schema_offset for index in question.question_span),
            option_spans=tuple(
                (start + schema_offset, end + schema_offset)
                for start, end in question.option_spans
            ),
            option_ids=question.option_ids,
        )
        for question in encoded_questions
    )

    input_ids = tuple(prefix_ids + state_ids + schema_ids + suffix_ids)
    return EncodedRecord(
        input_ids=input_ids,
        questions=shifted_questions,
        record_id=str(prompt.get("id", "")),
    )


def collate_prompts(
    records: list[EncodedRecord],
    *,
    pad_token_id: int,
) -> tuple[Any, Any]:
    """Pad records on the right for the schema head."""

    import torch

    if not records:
        raise ValueError("Supply at least one encoded record.")

    length = max(len(record.input_ids) for record in records)
    input_ids = torch.full((len(records), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(records), length), dtype=torch.long)

    for index, record in enumerate(records):
        record_length = len(record.input_ids)
        input_ids[index, :record_length] = torch.tensor(record.input_ids)
        attention_mask[index, :record_length] = 1

    return input_ids, attention_mask
