import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from audio_jev.text.model.lfm import Lfm2Model
from audio_jev.text.model.model import JevModel, JevModelConfig
from audio_jev.text.model.utilities import collate_prompts, encode_prompt
from audio_jev.text.train.lora import apply_lora
from audio_jev.text.train.loss import SchemaLoss


def encode_example(tokenizer, example, max_length):
    record = encode_prompt(tokenizer, example["prompt"], max_length=max_length)
    answers = example["answers"]
    question_ids = {question.question_id for question in record.questions}

    if set(answers) - question_ids:
        raise ValueError("An answer refers to an unknown question.")

    targets = []

    for question in record.questions:
        answer = answers.get(question.question_id)

        if answer is None:
            targets.append(None)
            continue

        if isinstance(answer, dict):
            if set(answer) != set(question.option_ids):
                raise ValueError("Soft targets must name every allowed option.")

            target = torch.tensor([answer[key] for key in question.option_ids], dtype=torch.float32)
        else:
            option = str(answer).lower() if isinstance(answer, bool) else str(answer)

            if option not in question.option_ids:
                raise ValueError(f"Unknown answer {answer!r} for {question.question_id}.")

            target = question.option_ids.index(option)

        SchemaLoss._distribution(target, torch.zeros(len(question.option_ids)))
        targets.append(target)

    if all(target is None for target in targets):
        raise ValueError("Each record must have at least one answer.")

    return record, targets


def read_dataset(path, tokenizer, max_length):
    examples = []

    with open(path) as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue

            try:
                examples.append(encode_example(tokenizer, json.loads(line), max_length))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

    if not examples:
        raise ValueError(f"The dataset is empty: {path}")

    return examples


def make_model(base_model, head_config=None, rank=8, alpha=16.0, revision=None):
    base = Lfm2Model.from_pretrained(base_model, revision=revision, dtype=torch.float32)
    config = JevModelConfig(base_model_config=base.config, **(head_config or {}))
    model = JevModel(config, base=base)
    apply_lora(model.base, rank, alpha)
    return model


def save_checkpoint(path, model, metadata):
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save({"metadata": metadata, "state": state}, path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    model = make_model(
        metadata["base_model"], metadata["head"], metadata["rank"],
        metadata["alpha"], metadata["revision"],
    )
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    if set(checkpoint["state"]) != expected:
        raise ValueError("Checkpoint parameters do not match the adapters and head.")

    model.load_state_dict(checkpoint["state"], strict=False)
    return model.to(device).eval()


def collate_examples(examples):
    records, targets = zip(*examples)
    return list(records), list(targets)


def run_epoch(model, loader, criterion, pad_token_id, device, optimizer=None, on_step=None):
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "cross_entropy": 0.0, "brier": 0.0}
    question_count = 0

    started = time.monotonic()

    for step, (records, targets) in enumerate(loader, 1):
        input_ids, attention_mask = collate_prompts(records, pad_token_id=pad_token_id)

        with torch.set_grad_enabled(optimizer is not None):
            logits = model(input_ids.to(device), records, attention_mask.to(device))
            result = criterion(logits, targets)

            if not torch.isfinite(result.loss):
                raise ValueError("The loss is not finite.")

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

        for name in totals:
            totals[name] += getattr(result, name).item() * result.question_count

        question_count += result.question_count

        if on_step is not None:
            on_step({
                **{name: getattr(result, name).item() for name in totals},
                "step": step,
                "questions": question_count,
                "elapsed_seconds": time.monotonic() - started,
            })

    if not question_count:
        raise ValueError("The loader has no labeled questions.")

    return {name: value / question_count for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train the Jev head and LFM LoRA adapters.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train")
    source.add_argument("--stream-open-jev", action="store_true")
    parser.add_argument("--validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--brier-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--train-records", type=int)
    parser.add_argument("--validation-records", type=int, default=32)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--trackio-project")
    parser.add_argument("--trackio-space")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.max_length) < 1:
        parser.error("Epochs, batch size, and maximum length must be positive.")

    if args.validation_records < 1 or args.shuffle_buffer < 1 or (
        args.train_records is not None and args.train_records < 1
    ):
        parser.error("Record limits and shuffle buffer must be positive.")

    if args.trackio_space and not args.trackio_project:
        parser.error("Set a Trackio project when using a Space.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    criterion = SchemaLoss(args.label_smoothing, args.brier_weight)
    model = make_model(args.base_model, rank=args.rank, alpha=args.alpha, revision=args.revision)
    revision = getattr(model.base.config, "_commit_hash", None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=revision)
    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if pad_token_id is None:
        raise ValueError("The tokenizer needs a padding or end token.")

    if args.stream_open_jev:
        from huggingface_hub import HfApi
        from audio_jev.text.train.streaming import OpenJevStream

        args.dataset_revision = HfApi().dataset_info(
            "ZefanCai/Open-Jev", revision=args.dataset_revision,
        ).sha
        train = OpenJevStream(
            tokenizer, "train", args.dataset_revision, args.max_length,
            args.train_records, args.shuffle_buffer, args.seed,
        )
    else:
        train = read_dataset(args.train, tokenizer, args.max_length)

    loader = DataLoader(
        train, batch_size=args.batch_size,
        shuffle=not args.stream_open_jev, collate_fn=collate_examples,
    )
    validation = None

    if args.validation:
        validation = DataLoader(
            read_dataset(args.validation, tokenizer, args.max_length),
            batch_size=args.batch_size, collate_fn=collate_examples,
        )
    elif args.stream_open_jev:
        validation = DataLoader(
            OpenJevStream(tokenizer, "validation", args.dataset_revision,
                          args.max_length, args.validation_records),
            batch_size=args.batch_size, collate_fn=collate_examples,
        )

    model.to(args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    metadata = {
        "base_model": args.base_model,
        "revision": revision,
        "rank": args.rank,
        "alpha": args.alpha,
        "head": {name: getattr(model.config, name) for name in (
            "width", "routing_layers", "layers", "heads", "feedforward", "dropout",
        )},
        "training": vars(args),
    }
    tokenizer.save_pretrained(output / "tokenizer")
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    tracker = None

    if args.trackio_project:
        import trackio

        tracker = trackio.init(
            project=args.trackio_project, name=args.run_name or output.name,
            space_id=args.trackio_space, private=True, config=metadata,
            auto_log_gpu=False, auto_log_cpu=False,
        )

    global_step = 0

    def log_step(values):
        nonlocal global_step
        global_step += 1
        values = {f"train/{name}": value for name, value in values.items()}

        if str(args.device).startswith("mps"):
            values["system/mps_driver_gb"] = torch.mps.driver_allocated_memory() / 1e9

        print(json.dumps({"global_step": global_step, **values}), flush=True)

        with (output / "metrics.jsonl").open("a") as destination:
            destination.write(json.dumps({"global_step": global_step, **values}) + "\n")

        if tracker is not None:
            tracker.log(values, step=global_step)

    for epoch in range(args.epochs):
        if args.stream_open_jev:
            train.epoch = epoch

        metrics = {"epoch": epoch + 1, "train": run_epoch(
            model, loader, criterion, pad_token_id, args.device, optimizer, log_step,
        )}

        if validation is not None:
            metrics["validation"] = run_epoch(
                model, validation, criterion, pad_token_id, args.device,
            )

        print(json.dumps(metrics), flush=True)
        save_checkpoint(output / f"epoch-{epoch + 1}.pt", model, metadata)

        if args.stream_open_jev:
            metrics["skipped_train_records"] = train.skipped
            metrics["skipped_validation_records"] = validation.dataset.skipped

        (output / f"epoch-{epoch + 1}.json").write_text(json.dumps(metrics, indent=2) + "\n")

        if tracker is not None:
            global_step += 1
            tracker.log({
                f"{split}/{key}": value
                for split in ("train", "validation") if split in metrics
                for key, value in metrics[split].items()
            }, step=global_step)

    if tracker is not None:
        tracker.finish()


if __name__ == "__main__":
    main()
