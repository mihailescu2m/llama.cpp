# llama.cpp — very large MoE models on a 64 GB Mac

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for running MoE models **far larger
than available RAM** by streaming their routed experts from SSD on demand, tuned specifically for
Apple Silicon.

Target hardware: **M1 Max, 64 GB, ~400 GB/s**. Everything except the routed experts stays resident;
the experts live in a bounded cache filled by demand loads and a one-layer-ahead prefetcher. The
usable checkpoint size is therefore set by **disk throughput, not by RAM** — a 284B model in 107 GiB
runs on a machine with 64 GB, at a speed that is genuinely usable for agentic coding.

Decode at depth is dominated by the KV read, not by weight traffic, which is why streaming the
experts costs so little once the context is deep. That is the entire premise of this approach, and
most of the optimisation effort targets prefill and long context rather than short-prompt decode.

---

## Support the Project

If this work is useful to you, a small donation is greatly appreciated and helps fund continued
development.

[![Donate with PayPal](https://www.paypalobjects.com/en_AU/i/btn/btn_donate_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=mihailescu2m%40gmail%2Ecom&lc=AU&item_name=memeka&item_number=odroid&currency_code=AUD&bn=PP%2DDonationsBF%3Abtn_donate_LG%2Egif%3ANonHosted)

---

## Building

Standard llama.cpp build; Metal is the only backend this fork is tuned for.

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build -j8 --config Release
```

Verify the ggml operations, including the ones this fork adds:

```bash
./build/bin/test-backend-ops -b MTL0 -o UNION_BUILD,FLASH_ATTN_UNION
```

The perf suite also carries expert-GEMM cases at both target models' shapes, which is what the
kernel tables in the research logs are generated from:

```bash
./build/bin/test-backend-ops perf -b MTL0 -o MUL_MAT
```

---

## Running

```bash
llama-server -m <first shard> \
  -ngl 99 --moe-stream --moe-stream-cache 40 --moe-stream-io-threads 8 \
  -c 131072 -b 4096 -ub 4096 -np 1 -fa on
```

Two parameters carry most of the performance:

* **`-ub 4096`** — dominates prefill, and
* **`--moe-stream-cache`** — size it to your machine's free RAM, *not* to the model. Leave ~4 GB of
  headroom or allocation fails, more with a draft model loaded.

Expert streaming is enabled by the CLI flag. Once it is on, lookahead prefetch and pair
partitioning are both **on by default**; the environment variables below exist to turn them off for
A/B work, and all of them parse their value, so `VAR=0` disables.

| Variable | Default | Effect |
|---|---|---|
| `LLAMA_MOE_STREAM_PARTITION` | on | give each (token, expert) pair to exactly one wave |
| `LLAMA_MOE_STREAM_LOOKAHEAD` | 1 | prefetch depth, in layers |
| `LLAMA_MOE_STREAM_CACHE` / `--moe-stream-cache` | — | expert cache budget, GiB |
| `LLAMA_MOE_STREAM_WAVE_CAP` | planner | force experts per wave; wins over the pair budget |
| `LLAMA_DSV4_UNION` | on | DeepSeek union-8; `0` selects the per-query sparse path |
| `LLAMA_QWEN4EXP_BLOCK_TOPK` | on | Qwen block-level indexer selection |
| `LLAMA_QWEN4EXP_INDEXER_F16` | on | keep raw indexer keys in F16 under quantised KV |
| `GGML_METAL_KPROF` | off | per-kernel GPU attribution, stride in nodes |
| `GGML_METAL_GPU_PROFILE` | off | per-context GPU busy time |

---

## What this fork adds on top of upstream

Organised by the commit layers in this branch. The reasoning behind each is in the model logs.

### Selected upstream PRs

Metal sparse flash attention, indexed predecessor lookup and focused Metal kernel improvements are
kept immediately above current llama.cpp master so they can be dropped when upstream merges them.

### MoE expert streaming

The core of the fork, and one commit per sub-feature so any of them can be dropped when upstream
grows an equivalent. Routed experts stream from SSD into a bounded cache: parallel slab reads at
real queue depth, one-layer-ahead prefetch on its own queue so a wide prefetch cannot delay a demand
read, route-hotness eviction with decay, GPU-side slot resolution serviced over a Metal shared event
(no CPU round-trip, no graph split), zero-copy loads straight into the Metal shared buffer, and
per-row streaming for gather tables too large to map.

Multi-pass prefill splits a ubatch that touches more experts than the cache holds into waves, and
pair partitioning gives each (token, expert) pair to exactly one wave rather than running every wave
over every pair and masking the rest away.

### Metal optimisations and profiling

Per-kernel GPU attribution behind `GGML_METAL_KPROF`, per-context GPU busy time, op-named debug
groups, an occupancy probe, and a routing-capture tool. On the kernel side: a flash-attention unroll
cap at DK=512, a byte-indexed half2 table for the MXFP4 GEMV, and naturally aligned halfword loads
for q8_0 dequant.

### DeepSeek

Union-8 lets blocks of eight prompt queries share one deduplicated top-k list, with each query's
exact membership preserved — so it is exact, not an approximation. Its threadgroup bitmap walks the
row space in chunks, so the path stays engaged at long context instead of falling back to dense.
Single-token decode never uses union-8.

### Qwen and serving

Block-level indexer selection cuts the prefill quadratic by selecting over blocks rather than
materialising an `[n_kv, n_tokens]` cell table. The indexer K cache stays F16 even under quantised
attention KV, because quantisation changes which blocks survive a discrete top-k. The native MTP
head is supported end to end, bounds its own memory, and can be gated by prompt length; saved slots
persist their prompt checkpoints.

---

## The two models, and why they are hard

They are hard in completely different ways, which is why each has its own log.

### DeepSeek-V4-Flash-0731 — the I/O problem

284B, 256 experts per layer. The model is 107 GiB and the machine has 64 GB, so the experts must
come off the disk *while the GPU waits*. Everything is about hiding that latency: prefetch far
enough ahead, keep the right experts resident, and never let a demand read queue behind speculative
work.

→ **[Research log](docs/DeepSeek-V4-Flash-0731.md)** — checkpoint composition, kernel survey, features, negative results.

### Qwen3.8-Flash-Next — the graph-shape problem

Fast enough that the bottleneck left the disk entirely. What remained was GPU work the graph did not
need to do: a reshape that silently broke Metal's `RMS_NORM→MUL` fusion, copies that bought nothing,
a full sort of 512 expert scores to read the top 10, and an indexer materialising a table
proportional to context × chunk size on every layer. Plus three genuinely awkward architectural
features: four parallel residual streams, a 26.8 GiB PLE table that cannot be resident, and a native
MTP head whose hidden-state contract has three separate ways to fail silently.

→ **[Research log](docs/Qwen3.8-Flash-Next.md)** — checkpoint composition, kernel survey, features, negative results.

---

## On the research logs

Both logs record **negative results as first-class content**, not as an appendix. Roughly half the
entries are ideas that look obviously correct on paper and cost real GPU time to disprove.

That is deliberate. On hardware this constrained, knowing which plausible optimisation *does not*
work — and why — has been worth more than the wins.

Each log also carries a **kernel survey**: every quant format the checkpoint could use, measured at
that model's own expert-GEMM shape, ranked by time per *effective* bit-per-weight. That ranking is
what selects a checkpoint's expert mix, and it does not match intuition — on this GPU the i-quant
kernels are occupancy-bound and lose to simpler formats that read more bytes.

---

## Upstream

This fork tracks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Upstream documentation
applies for everything not listed above; see [the upstream README](https://github.com/ggml-org/llama.cpp#readme)
for supported backends, model conversion and the general tool set.

Bugs found here that belong upstream are noted as such in the model logs.
