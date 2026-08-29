# Qwen3.8-Flash-Next (`qwen4exp`)

> Research log. Measured on **M1 Max, 64 GB, ~400 GB/s**, streaming experts from SSD.
> Conditions are quoted with every figure — in this log more than most, because several
> measurements here were later found to have been taken under conditions that invalidated them.

Where DeepSeek is a streaming problem, Qwen3.8-Flash-Next is a **graph-shape** problem. It decodes
fast enough that the bottleneck moved off the disk entirely and onto GPU work the graph did not
need to be doing: needless copies, broken kernel fusions, and an indexer that materialised a table
proportional to context x chunk size on every layer.

Three architectural features make it unusual, and all three cost real work to support:

- **Hyper-connections** — four parallel residual streams instead of one, mixed by a low-rank gate.
  Roughly 30% of decode GPU time, and the source of several fusion-breaking reshapes.
- **A 26.82 GiB PLE n-gram table** — far larger than RAM, touched a few rows at a time.
- **Gated DeltaNet on 36 of 48 layers**, with 12 full-attention layers carrying a sparse indexer.

---

## At a glance

| | |
|---|---|
| Layers | 48 — 36 SSM (gated DeltaNet) + 12 full attention |
| Experts | 512 per layer, 10 active per token |
| Attention | indexer `top_k = 2048`, `compress_ratio = 4`, head dim 256, GQA 48q:2kv |
| Hyper-connections | `hc = 4` streams |
| PLE table | 26.82 GiB, 90-byte rows — streamed, never resident |
| Checkpoint | `UD-iQ4_K_XXS`, 82.90 GiB, 3 shards (custom, see below) |
| Speculation | **native MTP head** (`mtp-…-Q4_0.gguf`, 2.2 GiB), n-max 4 |
| KV | F16 |

```bash
llama-server -m <first shard> \
  -ngl 99 --moe-stream --moe-stream-cache 32 --moe-stream-io-threads 8 \
  -c 131072 -b 4096 -ub 4096 -cms 4096 -np 1 \
  -md <mtp-…-Q4_0.gguf> \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 --spec-draft-ngl 99 \
  --spec-max-prompt 32768
```

`LLAMA_MOE_STREAM_LOOKAHEAD=1`. **Cache 32 with MTP, 36 without it.** The MTP head is resident on
top of the cache, and cache 36 + MTP *pages* — 1.1 GiB of swap and 5.3M swapouts, which invalidates
any measurement taken there. Cache 32 + MTP is clean.

The cache size is not free, which an earlier version of this log got wrong. See
[Cache size: 32 vs 36](#cache-size-32-vs-36).

---

## The checkpoint

`UD-iQ4_K_XXS` is a custom splice, built here rather than downloaded:

- base is unsloth's `UD-Q3_K_XL`;
- 43 of the 48 `ffn_down_exps` are replaced with **MXFP4** taken byte-for-byte from AtomicChat's
  `AD-3.84bpw-IQ4_XS-M64`, because MXFP4 is the format this fork has a hand-optimised Metal GEMV
  for. The five that unsloth deliberately kept at Q8_0 (layers 2, 4, 30, 46, 47) are left alone;
- `output.weight` is taken from `UD-Q4_K_XL` at Q8_0;
- the `IQ4_NL` PLE table is kept. AtomicChat ships it at `Q5_1`, which is 8.9 GiB larger, and that
  table is streamed per row - the last place to spend bytes.

The result is **0.9 GiB smaller than the Q3_K_XL it came from** despite the bigger output tensor.
Only the tensors named above differ from the base; 1180 of 1224 are untouched.

A trap worth recording: `--splice-type IQ4_NL` looks like the obvious selector for the
down-projections, and it also matches `per_layer_token_embd` — the PLE table is IQ4_NL too. With a
full donor that selector silently swaps the PLE table as well.

---

## Performance

### Where it started

Cache 36, ctx 65536, no speculation, 2 reps x 3 prompts:

| | t/s |
|---|---:|
| prefill (mean) | 121.10 |
| decode (128-tok mean) | 9.548 |

Both figures are from before any of the work below. Prefill reached 167.10 t/s partway through,
after the graph-shape fixes but before block-level top-k.

### Where it is now

`UD-iQ4_K_XXS`, cache 32, `-b 4096 -ub 4096 -fa 1`, llama-bench (`-r 2`, `-r 1` at 128k), no
speculation. Every cell verified swap-clean:

| context | pp t/s | tg t/s |
|---:|---:|---:|
| 4096 | 160.64 | 10.91 |
| 8192 | 159.11 | 10.82 |
| 16384 | 156.00 | 10.67 |
| 32768 | 150.81 | 10.02 |
| 65536 | 144.37 | 9.19 |
| 131072 | 132.79 | 8.03 |

Prefill falls 17.3% and decode 26.4% across a 32x context range.

### With MTP

Same model and cache, llama-server, one run per context, MTP as the only variable. Decode is the
*post-prefill* rate in both arms, so the two are directly comparable:

| context | pp no-MTP | pp MTP | tg no-MTP | tg MTP | decode gain |
|---:|---:|---:|---:|---:|---:|
| 4,061 | 159.14 | 150.11 | 10.44 | 17.55 | **+68%** |
| 8,307 | 157.03 | 146.91 | 10.70 | 20.47 | **+91%** |
| 16,649 | 155.63 | 147.25 | 10.32 | 19.34 | **+87%** |
| 33,339 | 151.43 | 141.51 | 10.35 | 19.12 | **+85%** |
| 66,505 | 142.62 | 131.97 | 9.38 | 16.57 | **+77%** |
| 129,983 | 131.45 | 119.37 | 8.60 | 12.99 | **+51%** |

**MTP is worth far more than the +39% this log used to record, and it still pays at 128k.** The
earlier figure was measured before the `t_h_nextn` export was fixed, when the head was running on a
hidden state that never reached it - see [Native MTP head](#native-mtp-head).

Prefill costs a steady 5.7-9.2%. End-to-end break-even, from these numbers:

| prompt | extra prefill | decode saved | repays after |
|---:|---:|---:|---:|
| 33k | 15.4 s | 0.0443 s/token | **~350 generated tokens** |
| 130k | 100.2 s | 0.0393 s/token | **~2,550 generated tokens** |

The 130k figure is close to the 2,369 measured previously, so the case for `--spec-max-prompt`
stands. The 32k break-even moved from 119 to ~350 tokens, so the threshold itself is worth
re-deriving rather than inherited.

### Cache size: 32 vs 36

An earlier version of this log said cache 32 was free, on the grounds that decode measured
identical at 32 and 36. **That was a decode-only measurement, and prefill was never tested.**
Interleaved A/B/A, two passes, same session, swap flat throughout:

| context | cache 36 | cache 32 | delta |
|---:|---:|---:|---:|
| 4096 | 186.46 / 186.06 | 160.57 / 160.53 | **-13.8%** |
| 8192 | 185.06 / 184.83 | 159.00 / 158.87 | **-14.1%** |
| 16384 | 180.87 / 181.22 | 156.03 / 156.05 | **-13.8%** |

**Cache 32 costs ~14% prefill and ~2% decode.** The penalty is flat across context, which is the
signature of a per-ubatch cost: every prefill ubatch sweeps the whole expert set, so 11% less
residency means ~14% more compulsory re-reads on each sweep. Prefill here is expert-I/O-bound.

That does not make 36 the right choice, because **cache 36 + MTP pages** (1.1 GiB swap, 5.3M
swapouts). The real choice is *cache 32 with MTP* against *cache 36 without it*, and MTP's +51-91%
decode dwarfs 14% of prefill for anything but a very long prompt with a very short answer.

### Prefill is a curve, not a number

The single most important result in this log. Block-level indexer top-k did not make prefill
uniformly faster — it flattened its **slope**:

| | ms/token |
|---|---|
| cell-level (before) | `5.828 + 3.864e-5·n` |
| block-level (after) | `6.026 + 6.668e-6·n` |

A **5.8x smaller quadratic coefficient**, at the cost of ~3% on the constant. Worth +14% at 32k,
**+71% at 160k**. Quoting this change as a single percentage is meaningless: measured at 32k it
looks like +18.5%, because at 32k the quadratic term is only 18% of the total.

---

## What was added, and why

### The indexer — the prefill story

| Change | Rationale | Measured |
|---|---|---|
| **Block-level top-k** | the budget is whole blocks anyway, so selecting over blocks skips materialising the `[n_kv, n_tps]` cell table entirely | **5.8x flatter prefill curve** |
| **Permute-free scoring** | putting `q` on the left lands the head index in dim 0, so `sum_rows` reduces heads directly — removing a permute+cont of ~640 MB per layer per 4096-token chunk | **+1.2% @ 32k, +1.4% @ 52k** |

Block-level selection is also a marginally *better* cut: all `r` cells of a block share a score, so
the cell-level top-k split the boundary block arbitrarily. Causality is unaffected — the causal
mask is still applied after selection.

### The graph — the decode story

| Change | Rationale | Measured |
|---|---|---|
| **RMS_NORM→MUL adjacency** | Metal's fusion check is a *tensor-identity* test; a reshape between the two silently broke it and spilled the norm to memory | part of +20.7% |
| **Hyper-connection collapse** | `ggml_cont` before the stream adds bought nothing — ADD needs contiguous *rows*, which a strided view already has — and blocked chain fusion | part of +20.7% |
| **`ggml_top_k` for MoE routing** | `argsort_top_k` fully sorted all 512 expert scores to view the first 10 | part of +20.7% |
| **Graph reuse / shared QSA input** | the whole graph was rebuilt and re-recorded every token | the largest share of +20.7% |

Combined A/B (cache 36, ctx 65536, no spec): **decode 9.548 → 11.522 t/s, +20.7%**; prefill flat.
Stream stats were unchanged between arms — miss 0.6–0.7%, stall 1.1%, GPU 98.7% — so the gain is
GPU work removed, not I/O.

### Native MTP head

The model ships a NextN block the loader read as part of the trunk and the graph could not run.
Supporting it needs five tensors beyond the shared NextN set, its own graph, and a hidden-state
export from the trunk.

**+51% to +91% decode** at n-max 4, depending on context — see [With MTP](#with-mtp).
This log previously recorded +39%; that was measured before the `t_h_nextn` export below was
fixed, i.e. while the head was still partly starved of the state it needs.

The head is fed the **wide** hyper-connection stream (`hc*n_embd`) from before the final mixer —
not the collapsed output — because its `hnorm` and `fc_hidden` are `[hc*n_embd]`. Three separate
things must hold or acceptance silently collapses to ~1% rather than failing loudly:

1. the trunk must set `res->t_h_nextn` at all (upstream `qwen4exp` sets only `t_embd`);
2. it must be the wide pre-mixer stream;
3. it needs an explicit `ggml_build_forward_expand` — it is a bare reshape view with no consumer,
   so otherwise the scheduler never assigns it a backend and it is never computed.

**The tell:** acceptance identical to the digit (8 accepted / 834 generated) across two different
drafter quantizations. Drafts that do not change when the drafter's weights change are not coming
from the drafter.

### PLE table streaming

26.82 GiB of 90-byte rows. Upstream keeps it off the resident set with `TENSOR_READ_LAZY`, which is
**mmap-only by construction** — rows arrive as page faults against a mapping the size of the whole
table. When expert streaming is on, the table is handed to the servicer instead: `set_input` reads
just this ubatch's rows with buffered parallel `pread`s into a compact tensor, and `get_rows`
dequantizes that exactly as it would the full table. Repeated rows are read once; reads are sorted
by file offset. Upstream's lazy path remains the fallback.

Deliberately **buffered, not `O_DIRECT`** — at 90 bytes a row the page cache genuinely helps, which
is the opposite of the finding for multi-MiB expert slabs.

### Correctness fixes

| Fix | Consequence if missing |
|---|---|
| **Recurrent-state rollback** | conv history and delta-net groups left zeroed/stale, corrupting generation after any rollback |
| **Indexer cache after sequence copies** | cached indexer keys are raw, so a pending update must not rope-shift them |
| **CPU-reference GQA head mapping** | `h % n_kv_head` instead of `h / (n_head/n_kv_head)` — made a *correct* kernel look broken on every real GQA shape |
| **`--spec-max-prompt`** | a draft head is an extra layer over the whole prompt; past a length it cannot pay back |

---

## Negative results

| Idea | Result |
|---|---|
| **MTP + n-gram stacked** | **net negative** — they compete for the same accepted tokens |
| **MTP at long context** | **inverts.** At 32k the head costs 3.7 s of prefill and repays after 119 generated tokens; at 128k it costs 57.7 s and repays after 2369. Hence the length gate. |
| **union-8 for QSA** | works, knee at kv≈20k — but block-topk got there first, and the payoff is small |
| **union-8 for MTP** | does not apply |
| **`hc_up` matmul at K=320** | a wash, not a win |
| **Upstream `5ea1b124` fa-vec tunings** | **rejected** — no F16 entry matches head dim 256/256; the one matching shape exists only for quantised KV, which is out |
| **Deeper drafts** | the expert GEMV n-curve is linear with a cliff at 32; depth does not amortise the weight read |

---

## Retractions

**"union-8 gives +4.3% / +6.1%."** Voided — measured on a **broken kernel**.

**"The dk256 union failure is a head-dimension problem."** Also wrong. The root cause was the
**CPU reference's** GQA head mapping; the Metal kernel had been correct all along.

**"Long-context decode collapses."** Wrong. The measurements behind it were 1–8 token generations
that hit EOS immediately. Measured properly with `ignore_eos`, decode at 122k is 7.13 (no spec) /
8.63 (MTP).

**"Block-topk is worth +18.5%."** Under-reported — measured at 32k, where the quadratic is only 18%
of prefill. At 128k it is worth far more.

**Upstream's `edb6dec1c` "enable recurrent state rollback" looks wrong**: it adds `QWEN4EXP` to the
whitelist, but their tree has neither the ring-bank conv writes nor the delta-net `n_written < K`
clamp, both prerequisites. Two upstreamable bug reports stand from this log — that one, and the
CPU-reference GQA mapping.

---

## Sweeps

| Sweep | Outcome |
|---|---|
| **ubatch / batch** | keep **4096** |
| **Cache size** | 36 is ~14% faster on prefill, ~2% on decode — but 36 + MTP pages. Use **32 with MTP** |
| **MTP quantization** | `Q4_0` (2.20 GiB); Q8_0 (3.85 GiB) gives similar acceptance (0.55 vs 0.51 mean) for 1.75x the footprint |
| **MTP n-max** | **4**; deeper drafts do not pay |
| **I/O threads with MTP** | keep **8** |
| **Speculation type x cache** | MTP alone beats MTP+n-gram and n-gram alone |

---

## Measurement discipline

- **The first request understates decode by ~25%.** Cold 12.20 → warm 16.49 / 16.26 t/s. Several
  single-shot figures in this log's history were cold and therefore pessimistic. Multi-arm sweeps
  issuing 3+ requests per arm are unaffected; one-shot verification runs are not.
- **That ~25% is a COLD-START effect, not a warm-vs-post-prefill one.** Measured directly, warm
  decode runs only 2-16% above post-prefill decode at the same depth. The two were conflated here
  previously. Post-prefill is the honest number for agentic use, where every turn re-prefills.
- **"It loaded" is not "it fits".** Cache 36 + MTP loads and runs, and pages while doing it. Sample
  `vm.swapusage` around every arm; a rising figure voids the measurement.
- **Warm decode measured once per context carries ~15% variance** — in one sweep the 8k warm figure
  came out *below* its own post-prefill figure, which is physically impossible. Prefer post-prefill
  for A/B work, or repeat the warm arm.
- **A replay trap can fake +33%.** n-gram speculation replaying its own prior output is not a
  measurement of anything.
- **Percentages on a curve are meaningless without the length.** See prefill, above.
- **Check the harness before the model.** A readiness loop waiting for the wrong log string looks
  exactly like a hung server. The line is `llama_server: listening on http://`.
- **Audit 3-way patch applies for silently reverted hunks** — conflict markers show only part of the
  damage; patch *context* can overwrite untouched work.

---

## Still open

- Generate F16 `256/256` flash-attention vector tunings — upstream never tuned this shape.
- The MTP head runs dense despite shipping sparse weights.
- Per-block bitmap for union-8 lifted its ceiling 4x, but no throughput gain at reachable contexts.
- Re-derive the `--spec-max-prompt` threshold: break-even moved from 119 to ~350 tokens at 32k now
  that MTP's decode gain roughly doubled.
- Separate how much of `UD-iQ4_K_XXS`'s prefill comes from MXFP4 versus from block-level top-k. No
  same-session A/B against `UD-Q3_K_XL` has been run.
