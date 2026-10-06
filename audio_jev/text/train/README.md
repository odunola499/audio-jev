# LoRA training

Run from the repository root with the project environment:

```sh
python -m audio_jev.text.train.trainer \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --output runs/lfm-lora-01 \
  --epochs 3 \
  --batch-size 2 \
  --device cuda
```

Use `--device cpu` for a CPU run or `--device mps` for an Apple GPU.
The output directory must be new. Validation is optional.

Each JSONL line contains a `prompt` accepted by `encode_prompt` and an `answers`
object keyed by question ID:

```json
{"prompt":{"state":"The customer asks for a refund.","questions":{"refund":{"type":"noul","instructions":"Does the customer want a refund?"},"team":{"type":"choice","criteria":{"billing":"Payment issues","support":"Technical issues"}},"urgency":{"type":"score","criteria":["Low","High"]}}},"answers":{"refund":true,"team":"billing","urgency":0}}
```

For `choice`, supply the option ID. For `score`, supply the zero-based level.
For `noul`, supply a Boolean or `"true"`/`"false"`. Soft targets are objects
mapping every option ID to a probability, for example `{"true":0.8,"false":0.2}`.
Omit a question or use `null` to skip its loss. Each record needs one labeled
question. Labels follow encoded option IDs, not dictionary insertion order.
Prompts longer than `--max-length` are rejected without truncation.

The default base is `LiquidAI/LFM2.5-350M`. Original parameters are frozen.
Rank-8 LoRA adapters with alpha 16 train on every backbone linear projection:
attention, short-convolution input/output projections, and feed-forward layers.
The convolution kernels, normalization weights, and token embeddings stay frozen.
The entire schema head trains. LoRA uses `W(x) + (alpha / rank) B(A(x))`, with
zero-initialized B. No extra dependencies are required.

This initial trainer uses float32, AdamW, gradient clipping at 1.0, and the
existing cross-entropy plus Brier loss. Metrics are averaged over labeled
questions. It runs on one device and saves after each epoch. It does not yet
include mixed precision, gradient accumulation, distributed training, or
optimizer-state resume. The prompt serializer is unchanged from `utilities.py`.

Each checkpoint contains only the adapters, head, and metadata. The tokenizer is
saved separately. Hub revision metadata records the resolved base commit when
available; local base directories must remain available and unchanged.

```python
from transformers import AutoTokenizer

from audio_jev.text.train.trainer import load_checkpoint

model = load_checkpoint("runs/lfm-lora-01/epoch-3.pt", device="cpu")
tokenizer = AutoTokenizer.from_pretrained("runs/lfm-lora-01/tokenizer")
```

The loaded model is in evaluation mode and accepts the same records, input IDs,
and attention mask as `JevModel`. Apply softmax to each returned logit vector
to obtain option probabilities.

## Open-Jev smoke test

Download the raw `release-v2-redistributable` train and validation splits from
`ZefanCai/Open-Jev`, then convert each split separately:

```sh
python -m audio_jev.text.train.open_jev \
  data/open-jev-source/raw/release-v2-redistributable/train.jsonl.gz \
  data/open-jev-smoke/train.jsonl --limit 4

python -m audio_jev.text.train.open_jev \
  data/open-jev-source/raw/release-v2-redistributable/validation.jsonl.gz \
  data/open-jev-smoke/validation.jsonl --limit 2

python -m audio_jev.text.train.trainer \
  --train data/open-jev-smoke/train.jsonl \
  --validation data/open-jev-smoke/validation.jsonl \
  --output runs/open-jev-mps-smoke \
  --epochs 2 --batch-size 1 --device mps --max-length 4096
```

The converter groups rows by group ID and exact state, preserves soft labels,
and explicitly maps Boolean option names. Choice IDs are source option indices;
their full descriptions are preserved. Label-bearing metadata is not included
in the model prompt. Omit `--limit` to convert the whole split. Limits select
the first complete records, so this small sample is not representative of the
whole dataset. The recorded smoke test used customer-control examples only.
