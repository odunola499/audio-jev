import argparse
import gzip
import json
from pathlib import Path


def convert_rows(rows):
    """Combine fields with the same group and state. Keep probability targets."""

    groups = {}

    for row in rows:
        key = (row["group_id"], json.dumps(row["state"], sort_keys=True))
        example = groups.setdefault(key, {
            "prompt": {"id": row["group_id"], "state": row["state"], "questions": {}},
            "answers": {},
        })
        question_id = row["id"]

        if question_id in example["answers"]:
            raise ValueError(f"Duplicate question: {question_id}")

        options = row["options"]
        target = row["target"]

        if len(options) != len(target) or not options:
            raise ValueError("Each option needs a target probability.")

        kind = row["kind"]

        if kind == "noul":
            aliases = {"no": "false", "yes": "true", "false": "false", "true": "true"}
            option_ids = [aliases[option.lower()] for option in options]

            if set(option_ids) != {"true", "false"} or len(option_ids) != 2:
                raise ValueError("Boolean options must contain one true and one false value.")

            criteria = dict(zip(option_ids, options))
        elif kind in {"choice", "score"}:
            option_ids = [str(index) for index in range(len(options))]
            criteria = dict(zip(option_ids, options)) if kind == "choice" else options
        else:
            raise ValueError(f"Unknown question type: {kind}")

        example["prompt"]["questions"][question_id] = {
            "type": kind,
            "instructions": row["question"],
            "criteria": criteria,
        }
        example["answers"][question_id] = dict(zip(option_ids, target))

    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description="Convert an Open-Jev raw split to training JSONL.")
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("The record limit must be positive.")

    with gzip.open(args.source, "rt") as source:
        examples = convert_rows(json.loads(line) for line in source if line.strip())

    examples = examples[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x") as destination:
        for example in examples:
            destination.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(json.dumps({"records": len(examples), "questions": sum(
        len(example["answers"]) for example in examples
    )}))


if __name__ == "__main__":
    main()
