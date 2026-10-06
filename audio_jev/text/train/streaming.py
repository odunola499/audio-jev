import itertools
import json
import random

from datasets import load_dataset
from torch.utils.data import IterableDataset

from audio_jev.text.model.utilities import encode_prompt
from audio_jev.text.train.open_jev import convert_rows


def grouped_examples(rows):
    """Collect each source block before joining fields with the same state."""

    seen_sources = set()

    for source, block in itertools.groupby(rows, key=lambda row: row["source"]):
        if source in seen_sources:
            raise ValueError("Source blocks must be contiguous before grouping.")

        seen_sources.add(source)
        yield from convert_rows(block)


def shuffle_buffer(examples, size, seed):
    rng = random.Random(seed)
    buffer = []

    for example in examples:
        if len(buffer) < size:
            buffer.append(example)
            continue

        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = example

    rng.shuffle(buffer)
    yield from buffer


class OpenJevStream(IterableDataset):
    def __init__(self, tokenizer, split, revision, max_length, limit=None,
                 shuffle_size=0, seed=42):
        self.tokenizer = tokenizer
        self.split = split
        self.revision = revision
        self.max_length = max_length
        self.limit = limit
        self.shuffle_size = shuffle_size
        self.seed = seed
        self.epoch = 0
        self.skipped = 0

    def __iter__(self):
        from audio_jev.text.train.trainer import encode_example

        dataset = load_dataset(
            "ZefanCai/Open-Jev", "release-v2-redistributable",
            split=self.split, revision=self.revision, streaming=True,
        )
        examples = grouped_examples(json.loads(row["record_json"]) for row in dataset)

        if self.shuffle_size:
            examples = shuffle_buffer(examples, self.shuffle_size, self.seed + self.epoch)

        self.skipped = 0
        count = 0

        for example in examples:
            record = encode_prompt(self.tokenizer, example["prompt"])

            if len(record.input_ids) > self.max_length:
                self.skipped += 1
                continue

            yield encode_example(self.tokenizer, example, self.max_length)
            count += 1

            if self.limit is not None and count >= self.limit:
                break
