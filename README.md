# lora-experiments

LoRA fine-tuning of a 1.5B model for structured JSON output, measured against
prompting baselines on a held-out test set.

The interesting part is not that fine-tuning worked. It is which of the usual
knobs turned out not to matter, and the one number that showed a tuned adapter
must never be served unmerged.

**Model:** Qwen2.5-1.5B-Instruct · **Hardware:** RTX 3090 · **Training time:** 2 min/run
**Stack:** TRL SFTTrainer, PEFT, transformers

---

## Headline results

A 1.5B model trained on 400 examples matches a 3B model with a fully
engineered prompt — while using a prompt 13× shorter:

| configuration | prompt | exact | json | schema | enum |
|---|---|---|---|---|---|
| 1.5B, engineered prompt | 190 tok | 0.100 | 0.98 | 0.98 | 0.93 |
| 3B, engineered prompt | 190 tok | 0.150 | 1.00 | 0.93 | 0.90 |
| **1.5B + LoRA** | **15 tok** | **0.150** | **1.00** | **1.00** | **1.00** |

The tuned model is the only configuration where all 60 test outputs are
schema-valid and enum-valid. The 3B model with a full prompt is not.

Three findings, in order of how much they changed my mental model:

| finding | measurement |
|---|---|
| **rank doesn't matter** | r=4 (0.3% of params) == r=32 (2.3%) |
| **an unmerged adapter costs half your throughput** | 37.2 → 18.1 tok/s; merge restores it exactly |
| **loss is not the metric** | 12× lower eval loss bought one test example |

---

## What the task is

Input: a product review. Output: one JSON object, no fences, no prose.

```json
{"stars": 4, "sentiment": "positive", "topics": ["delivery", "quality"], "recommend": true}
```

Labels come from a public review dataset (`SetFit/amazon_reviews_multi_en`),
stratified 120 per star rating. `stars` is the real human rating; `sentiment`
and `recommend` are deterministic functions of it; `topics` comes from a
keyword lexicon.

**This is a deliberate trade-off and it has a cost.** Three of four fields are
derived, so what the metric really measures is `stars` × `topics`. The
alternative — hand-labelling 400 examples to carry conventions that exist
only in my head — was measured and rejected at four hours of work. The 60
hand-labelled examples from that attempt are kept in `data/test_manual_4fields.jsonl`.

---

## Finding 1 — rank is not the knob

Four ranks, same data, same seed, one sweep:

| rank | trainable params | % of model | exact | stars | topics |
|---|---|---|---|---|---|
| 4 | 4,616,192 | 0.298% | 0.150 | 0.467 | 0.267 |
| 8 | 9,232,384 | 0.595% | 0.150 | 0.467 | 0.233 |
| 16 | 18,464,768 | 1.182% | 0.150 | 0.467 | 0.250 |
| 32 | 36,929,536 | 2.336% | 0.150 | 0.500 | 0.283 |

`r=4` trains 0.3% of the model and matches `r=32` on every aggregate metric.
This is the low intrinsic rank hypothesis from the LoRA paper, reproduced:
adapting a model to a narrow task does not require many independent directions.

A note on reading this table. Identical `exact` across four rows looks like a
bug — a constant that didn't get wired through. It isn't: the per-field numbers
*do* differ (`stars` 0.467 vs 0.500, `topics` 0.233 vs 0.283). The models are
genuinely different. What's identical is how often all four fields land at once,
and that is a property of the metric, not the models.

---

## Finding 2 — never serve an unmerged adapter

batch=1, 100 new tokens, best of 5 runs:

| configuration | tok/s | |
|---|---|---|
| base | 37.2 | |
| base + adapter | 18.1 | **−51%** |
| merged | 37.4 | identical to base |

The paper's claim that merging eliminates inference overhead holds exactly.
The surprise is the size of the penalty *before* merging: half the throughput,
from 1.2% more parameters.

The arithmetic explains it. LoRA adds two small matmuls per adapted projection —
7 projections × 28 layers ≈ 400 extra kernel launches per token. The FLOPs are
negligible; the launches are not. At batch=1 decode is bound by launch overhead,
not compute, so a change that adds almost no work still costs half the speed.

**Merging is not bit-exact.** The merged model scored `stars` 0.483 vs the
adapter's 0.467 and `topics` 0.267 vs 0.250 — one example each. `W + BA·(α/r)`
is computed and stored in bf16, which has 8 mantissa bits, and rounding flips
borderline cases. A merged model that disagrees slightly with its adapter is
expected behaviour, not a bug.

---

## Finding 3 — the loss fell 12× and bought one example

The first training run computed loss over the entire sequence, review text
included. `eval_loss` sat at 1.287 and barely moved across epochs — most of
the gradient signal was going into predicting natural language that cannot be
predicted. The JSON was 20 tokens out of 150.

Switching to `assistant_only_loss=True`:

| | eval_loss | exact |
|---|---|---|
| loss over full sequence | 1.287 | 0.133 |
| loss over completion only | 0.110 | 0.150 |

The two losses are not directly comparable — they are computed over different
token sets, and predicting rigid JSON is easy in a way predicting prose is not.
But that makes the conclusion stronger, not weaker: a genuinely better gradient
still bought one test example out of sixty, because the ceiling was somewhere
else entirely.

---

## Where the ceiling actually is

`stars` sits at 0.45–0.50 in **every** configuration — base 1.5B, base 3B, and
all five adapters. Nothing moves it. Error distribution for the tuned model:

```
predicted − actual:   -2: 5    -1: 19    0: 28    +1: 7    +2: 1

exact match      0.467
within ±1 star   0.900
```

The model reads tone correctly and cannot recover the exact rating, because
people do not assign stars consistently. There is a systematic bias too:
`-1` occurs 19 times against 7 for `+1` — the model under-rates. Real reviewers
give higher ratings than their own text implies (*"disappointed but not worth
returning"* → 4 stars).

**So `exact` was the wrong metric.** It demands hitting four fields at once,
one of which has a noise ceiling near 0.5. No amount of training moves it. A
per-field report, or `stars_within_1`, would have been honest. This is a flaw
in how I set up the measurement, not in the model — and it is why the rank
ablation above reads as a flat line.

The `topics` lexicon has its own ceiling: it fires on substrings, so
*"worth saying something"* → `price`, and *"not worth returning"* → `support`,
which is the opposite of contacting support. The model learns the lexicon
faithfully, errors included.

---

## Running it

```bash
pip install "transformers>=4.46" "trl>=0.12" peft accelerate datasets

python -m src.fetch_reviews        # stratified sample, 120 per star
python -m src.build_dataset        # → data/{train,val,test}.jsonl
python -m src.to_chat              # → chat-format JSONL for TRL

python train_lora.py 4             # rank as argument → out/lora-r4
python infer_eval.py out/lora-r4/final
python merge_bench.py              # merge + latency comparison
```

Baselines against Ollama, for comparison with the prompting path:

```bash
python -m src.run_baseline qwen2.5:1.5b-instruct-q4_K_M v3
python -m src.run_baseline qwen2.5:3b-instruct-q4_K_M v3
```

### Training configuration

LoRA on all seven projections (`q,k,v,o,gate,up,down`), `alpha = 2r`,
dropout 0.05, lr 2e-4 cosine, 5 warmup steps, effective batch 16, 3 epochs,
bf16, `assistant_only_loss=True`. 75 steps, about two minutes on a 3090.

---

## Layout

```
src/
  prompts.py        prompt constants + deterministic labelling rules
  fetch_reviews.py  stratified sampling from the HF dataset
  build_dataset.py  texts + stars → labelled JSONL
  to_chat.py        → chat format for TRL
  eval_json.py      strict parser, 4 metrics, per-field breakdown
  run_baseline.py   prompting baselines via Ollama
train_lora.py       LoRA training, rank as CLI argument
infer_eval.py       generation + evaluation for an adapter
merge_bench.py      merge_and_unload + latency benchmark
data/
  test.jsonl                   60 held-out examples
  test_manual_4fields.jsonl    60 hand-labelled, earlier schema
```

A strict parser matters here: no stripping of markdown fences, no regex
extraction of the first `{...}`. Repairing the output would hide exactly the
behaviour the training is meant to fix.

---

## Known limitations

- **60 test examples.** A difference of 0.017 is one example. Nothing in the
  rank ablation should be read as signal.
- **Greedy decoding, single seed per rank.** No variance estimate across
  training seeds, so small per-field differences are not separable from noise.
- **Three of four fields are derived** from `stars`, which inflates the
  apparent agreement between fields.
- **`topics` measures lexicon recall**, not understanding.

## Related

[rag-docs](https://github.com/HoLokostikk/rag-docs) — retrieval and generation
evaluated end to end · [llm-playground](https://github.com/HoLokostikk/llm-playground) —
inference benchmarks; the kernel-launch explanation for the unmerged-adapter
penalty comes from GPU measurements there.
