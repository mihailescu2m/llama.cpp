# DeepSeek-V4-Flash-0731

> Research log. Every number here was measured on **M1 Max, 64 GB, ~400 GB/s**, against an SSD.
> Conditions are quoted with each figure because most of them do not survive being quoted without.

A 284B mixture-of-experts model on a machine with 64 GB of RAM. The routed experts — the
overwhelming majority of the weights — never become resident. They are streamed from SSD on
demand into a bounded cache, so the usable checkpoint size is set by **disk throughput, not by
RAM**.

Decode at depth is dominated by the KV read rather than by weight traffic, which is why streaming
the experts costs so little once the context is deep. That is the premise of this work, and the
reason the model runs well especially at larger contexts — most of the optimisation effort here
targets prefill and long context rather than short-prompt decode.

---

## The checkpoint

The default is a **custom quant**, `UD-iQ4-XXS`, built for this machine:

```
/Users/marianmi/Local/llm/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-UD-iQ4-XXS-00001-of-00004.gguf
```

It exists because the off-the-shelf quants optimise for size at the expense of the one thing that
matters here: **which Metal kernels the expert tensors end up using**. This mix deliberately favours
quant types whose kernels are fast on M1 — the i-quant kernels are occupancy-bound on this GPU and
lose badly to the simpler formats regardless of how few bytes they read — while keeping
KL-divergence against the full-precision reference very low. It is larger on disk than the stock
3-bit quants and faster in practice, which is the right trade when the weights are streamed anyway
and the bottleneck is kernel throughput rather than file size.

---

## At a glance

| | |
|---|---|
| Parameters | 284B MoE, 43 layers, 256 experts/layer, 6 active per token |
| Attention | MLA + DSA sparse attention, indexer `top_k = 512` |
| Checkpoint | `UD-iQ4-XXS`, 107.34 GiB, 4 shards (custom, see above) |
| Resident | everything except routed experts |
| Streamed | routed experts, into a 40 GiB cache |
| Speculation | n-gram (`ngram-mod`) — model-free, costs no RAM |
| KV | F16 (quantised KV is a standing constraint, see [Negative results](#negative-results)) |

```bash
llama-server -m <first shard> \
  -ngl 99 --moe-stream --moe-stream-cache 40 --moe-stream-io-threads 8 \
  -c 131072 -b 4096 -ub 4096 -np 1 -fa on \
  --spec-type ngram-mod --spec-ngram-mod-n-match 16 \
  --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 16
```

`--moe-stream-cache 40` is the measured optimum on 64 GB — **size it to your machine, not to the
model**.

`-ub 4096` is the single largest prefill parameter.

Expert streaming, union-8 attention and the lookahead prefetcher are on by default; each has a kill
switch (`LLAMA_MOE_STREAM_LOOKAHEAD=0`, `LLAMA_DSV4_UNION=0`).

---

## Performance

### Where it started

| | t/s |
|---|---:|
| prefill | 32 |
| decode, warm | 6.68 |

### Where it is now

`UD-iQ4-XXS`, cache 40 GiB, `-b 4096 -ub 4096 -fa 1 -r 1`:

| context | pp t/s | tg t/s | peak mem |
|---:|---:|---:|---:|
| 8192 | `<?>` | `<?>` | `<?>` |
| 32768 | `<?>` | `<?>` | `<?>` |
| 65536 | `<?>` | `<?>` | `<?>` |
| 131072 | `<?>` | `<?>` | `<?>` |

> **Pending re-measurement.** The previous figures in this table — pp 100.30 / 92.33 / 82.52 and
> tg 7.31 / 7.55 / 7.30 at 32k / 64k / 128k — were taken on the older `UD-IQ3_XXS` checkpoint at
> cache 44 GiB, so they do not describe the configuration above and have been removed rather than
> relabelled.

Results obtained running:

```bash
llama-bench -m <model> -ngl 99 --moe-stream --moe-stream-cache 40 \
  --moe-stream-io-threads 8 -b 4096 -ub 4096 -fa 1 -r 1 \
  -p 8192,32768,65536,131072 -n 0
```

`-r 1` means each cell is a single sample against a ~3.5% machine spread. Decode figures from
llama-bench are *post-prefill*, the regime agentic use actually sees; warm decode after several
hundred tokens is higher.

---

## What was added, and why

### Attention and prefill

| Change | Rationale | Measured |
|---|---|---|
| **Cap the flash-attention unroll at DK=512** | the heuristic `MIN(DK8/2, 4*NSG)` yields 32 at DK=512, spilling registers | **prefill +60–90%** |
| **Union-8 sparse attention** | 8 queries share one deduplicated top-k list instead of 8 separate gathers | **1.94x over dense at kv=16384**, flat in context |
| **Pair-partitioned waves** | expert GEMMs run over a partitioned pair list, not one expert at a time | **prefill +37%** |

Union-8 is the reason decode stays flat as context grows. It also produced this log's most
instructive bug: see [Retractions](#retractions).

### MoE expert streaming

| Change | Rationale | Measured |
|---|---|---|
| **Lookahead prefetch** | issue layer *n+1*'s expert loads during layer *n*'s compute | **+14.5% decode**, −52% misses |
| **Hotness decay 64 → 1024 tokens** | eviction was forgetting the working set faster than it was established | **+17.6%** |
| **Parallel slab reads** | one expert's weight slabs were read sequentially at QD1 | **+12%** |
| **Hash-layer prefetch** | route-hash prediction for the next layer | **+2.5%**, −29% misses |
| **Zero-copy expert load** | reads land straight in the Metal shared buffer, no staging copy | removes a full copy per miss |
| **GPU-side slot resolve** | the cache-slot lookup ran on CPU, forcing a graph split per layer | 0 mismatches over 11,798 calls |

Stall is **per layer, not per miss** — several misses inside one layer cost roughly what one does.
That single fact explains why prefetch width is cheap and why miss *count* is a poor proxy for time.

### Speculation and serving

| Change | Rationale | Measured |
|---|---|---|
| **n-gram speculation** (`ngram-mod`, n_max 16) | zero RAM, and RAM is the scarce resource here | **+5.2%** |
| **Draft context sized for the draft model** | an MoE drafter inherited the target's batch and cache budget — 11–13 GB for a 7.2 GB file | frees the memory speculation exists to pay for |
| **Slot checkpoint persistence** | a restored slot re-prefilled from scratch on an SWA model | 3804 tokens/40.0 s → **4 tokens/1.2 s** |

---

## Negative results

The more useful half of the log. Several of these look obviously correct on paper and cost real GPU
time to disprove.

| Idea | Result |
|---|---|
| **Depth-2 lookahead** | **−4.7%.** Failed its own 3.75% gate. Depth-1 is the whole win. |
| **i-quant LSU wide loads** | **−19 to −28%.** The q4_0 gap is *occupancy*, not load width. |
| **MoE small-batch kernel** | wins 17% synthetically, **loses 5.9% in situ** — and hid an `ne11==1` broadcast bug a unit test cannot catch |
| **Scan resistance / tail-hotness** | built, measured worse, shipped off. Prefill is 1.6% of the working set; there is nothing to resist. |
| **Page-cache L2** | page cache serves ~3% of expert reads. All `F_NOCACHE`/L2 work is dead. |
| **Removing graph splits** | removed 97% of them for **zero gain** |
| **Block-sparse / FA block-skip** | both measured dead |
| **Quantised KV** | out by constraint, not by measurement |
| **ANE / `lm_head` offload** | declined — no path to a win worth the complexity |

**Standing rule: no output-altering optimizations.** Quality is not tradeable for single-digit
percentage gains, which retires a whole class of otherwise attractive ideas.

---

## Retractions

Claims this log made and later withdrew. They are recorded because each one was believed, acted on,
and wrong — and because the *reason* each was wrong is the reusable part.

**"There is a cache-coverage knee at 43%."** There is no knee. The apparent knee was a **drafter
confound**; headroom sets cache size, nothing else does.

**"i-quant kernels carry a decode penalty, so requantize to Q3_K."** Backwards. Q3_K measured **29%
slower**. Do not requantize an i-quant to a k-quant expecting a speedup — the win comes from picking
formats whose kernels suit the GPU, which is what the custom checkpoint above does.

**"The drafter costs a third of every pass."** Corrected: drafter GPU time is only **~9 ms/pass**.
The 61 ms delta was almost entirely I/O.

**"Graph splits are the decode target."** They are not — removing 97% of them changed nothing.
Decode with no expert I/O at all is 16.65 t/s, which is the real ceiling.

**Inherited error:** the upstream project this work started from states its cache-coverage figure
backwards. Do not inherit its "more cache buys nothing" conclusion.

---

## Sweeps

| Sweep | Outcome |
|---|---|
| **Expert cache size** | optimum **40 GiB** on 64 GB; leave ~4 GB headroom or allocation fails |
| **I/O threads** | 8; the QD1 problem is fixed and what remains is structural and small |
| **ubatch / batch** | 4096 — the single largest prefill parameter |
| **Expert GEMV n-curve** | linear with a cliff at 32 — deep drafts do not amortise the weight read, so cap draft depth at 24 |

---

## Measurement discipline

Traps this log fell into, so they can be avoided rather than rediscovered.

- **Output is not reproducible across processes.** Text diffs cannot verify output safety; use an
  in-graph invariant instead.
- **Decode t/s is a bad A/B metric for I/O changes.** Prefill sweeps the whole expert set into the
  cache and evicts the decode working set, so decode straight after a prefill is roughly half the
  warm rate and recovers over several generations. Quote warm, post-prefill and blended figures
  separately, and never against each other.
- **Never leave another GPU job running.** A stray `test-backend-ops` produced 4.33 vs 10.35 t/s for
  *identical* configurations.
- **A hand-rolled benchmark arm can measure half the real work.** The partitioning env var is a
  known trap of exactly this kind.
- **macOS `bash` 3.2 treats an empty array as unbound** — it has silently killed the control arm of
  a sweep twice.

---

## Still open

- Re-measure the results table above on `UD-iQ4-XXS` at cache 40.
- Generate F16 `512/512` flash-attention vector tunings; upstream never tuned this shape, so the
  kernel falls back to a heuristic.
- The decode ceiling with no expert I/O is 16.65 t/s against ~70 of 400 GB/s achieved — the GPU-side
  gap is the remaining structural headroom, and neither graph reuse nor Metal splits close it.
