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

## Benchmarks

`tools/llama-bench` is updated to support MoE streaming (`--moe-stream`, `--moe-stream-cache`,
`--moe-stream-io-threads`), so every figure below is reproducible with the stock harness.

### Qwen3.8-Flash-Next — `UD-iQ4_K_XXS` (82.90 GiB), cache 32 GiB

llama-bench, no speculation, `-r 2` (`-r 1` at 128k):

| context | pp t/s | tg t/s |
|---:|---:|---:|
| 4096 | 160.64 | 10.91 |
| 8192 | 159.11 | 10.82 |
| 16384 | 156.00 | 10.67 |
| 32768 | 150.81 | 10.02 |
| 65536 | 144.37 | 9.19 |
| 131072 | 132.79 | 8.03 |

With the native MTP head, decode rises to **17.6 t/s at 4k and 13.0 t/s at 128k** — a gain of
**+51% to +91%** depending on context, for 5.7–9.2% of prefill. Full A/B in the model log.

### DeepSeek-V4-Flash-0731 — `UD-iQ4-XXS` (107.34 GiB), cache 40 GiB

| context | pp t/s | tg t/s |
|---:|---:|---:|
| 4096 | 56.28 | 6.86 |
| 8192 | 55.11 | 6.66 |
| 16384 | 53.02 | 6.52 |
| 32768 | 50.69 | 6.68 |
| 65536 | 47.62 | `<?>` |
| 131072 | `<?>` | `<?>` |

The DeepSeek sweep was stopped before the two longest contexts. Its prefill is also well below what
this checkpoint's predecessor recorded (`UD-IQ3_XXS` at cache 44 measured pp32768 = 100.3), and that
gap is **unexplained** — different checkpoint, different cache size and a possible regression are all
still on the table. Treat the DeepSeek column as provisional until that is resolved.

Usage:

```bash
llama-bench -m <model> -ngl 99 --moe-stream --moe-stream-cache 32 \
  --moe-stream-io-threads 8 -b 4096 -ub 4096 -fa 1 -r 2 \
  -p 4096,8192,16384,32768,65536,131072 -n 0
```

Decode figures are *post-prefill* — the regime agentic use actually sees, since every turn
re-prefills. Warm decode runs 2–16% higher. Every cell above was checked for paging by sampling
`vm.swapusage` around it; a configuration that swaps produces numbers that look fine and mean
nothing.

---

## Building

Standard llama.cpp build; Metal is the only backend this fork is tuned for.

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build -j8 --config Release
```

Verify the ggml operations, including the ones this fork adds:

```bash
./build/bin/test-backend-ops -o UNION_BUILD -o FLASH_ATTN_UNION
```

---

## Running

```bash
llama-server -m <first shard> \
  -ngl 99 --moe-stream --moe-stream-cache 40 --moe-stream-io-threads 8 \
  -c 131072 -b 4096 -ub 4096 -np 1 -fa on
```

Two parameters carry most of the performance:
* **`-ub 4096`** - dominates prefill, and
* **`--moe-stream-cache`** - should be sized to your machine's free RAM, *not* to the model — leave
~4 GB of headroom or allocation fails, more if using MTP.

Streaming, union-8 attention and the lookahead prefetcher are on by default, each with a kill switch.

---

## What this fork adds on top of upstream

Organised by the commit layers in this branch. The measured effect of each is in the model logs.

* **MoE expert streaming**
The core of the fork. Routed experts stream from SSD into a bounded cache: parallel slab reads,
one-layer-ahead prefetch, route-hotness eviction with decay, GPU-side slot resolution (no CPU
round-trip, no graph split), zero-copy loads straight into the Metal shared buffer, and per-row
streaming for gather tables too large to map.

* **Metal optimisations**
Per-kernel GPU attribution behind `GGML_METAL_KPROF`, per-context GPU busy time with op-named debug
groups, an occupancy probe, and a routing-capture tool. A flash-attention unroll cap at DK=512
(worth **+60–90% prefill**) and a byte-indexed half2 table for the MXFP4 GEMV. KPROF is the reason
most of the rest of this list exists — it is what turned "decode feels slow" into a ranked list.

* **Union-8 sparse attention**
Eight queries share one deduplicated top-k list instead of eight independent gathers — **1.94x over
dense at kv=16384**, and the reason decode stays flat as context grows.

* **Speculation & serving**
Model-free n-gram speculation (DeepSeek), a native MTP head (Qwen) with `--spec-max-prompt` to stop a draft head from costing more prefill than it saves, correct draft-context sizing for MoE drafters, and slot
checkpoint persistence so a restored session does not re-prefill from scratch.

* **Model support & fixes**
Recurrent-state rollback, indexer-cache correctness across sequence copies, graph-shape fixes that
restore broken kernel fusions, and block-level indexer top-k that flattens the prefill quadratic by
**5.8x**.

---

## The two models, and why they are hard

They are hard in completely different ways, which is why each has its own log.

### DeepSeek-V4-Flash-0731 — the I/O problem

284B, 256 experts per layer. The model is 107 GiB (`UD-iQ4-XXS`) and the machine has 64 GB, so the experts must
come off the disk *while the GPU waits*. Everything is about hiding that latency: prefetch far
enough ahead, keep the right experts resident, and never let a demand read queue behind speculative
work. The instructive part is how much of the obvious tuning turned out to be wrong — the page
cache serves 3% of reads, removing 97% of graph splits changed nothing, and stall is per *layer*,
not per miss.

→ **[Full research log](docs/DeepSeek-V4-Flash-0731.md)** — features, negative results, retractions, sweeps.

### Qwen3.8-Flash-Next — the graph-shape problem

Runs on `UD-iQ4_K_XXS`, a custom splice built here: unsloth's `UD-Q3_K_XL` with 43 of its 48
down-projections swapped to **MXFP4** — the one expert format this fork has a hand-optimised Metal
GEMV for — while keeping the smaller `IQ4_NL` PLE table. 0.9 GiB smaller than the base it came from.

Fast enough that the bottleneck left the disk entirely. What remained was GPU work the graph did not
need to do: a reshape that silently broke Metal's `RMS_NORM→MUL` fusion, copies that bought nothing,
a full sort of 512 expert scores to read the top 10, and an indexer materialising a table
proportional to context x chunk size on every layer. Plus three genuinely awkward architectural
features: four parallel residual streams, a 26.8 GiB n-gram table that cannot be resident, and a
native MTP head whose hidden-state contract has three separate ways to fail silently.

→ **[Full research log](docs/Qwen3.8-Flash-Next.md)** — features, negative results, retractions, sweeps.

---

## On the research logs

Both logs record **negative results and retractions as first-class content**, not as an appendix.
Roughly half the entries are ideas that look obviously correct on paper and cost real GPU time to
disprove; several more are claims this project made, believed, acted on, and later had to withdraw.

That is deliberate. On hardware this constrained, knowing which plausible optimisation *does not*
work — and why — has been worth more than the wins. Each retraction records the reasoning error, not
just the corrected number, because the error is the reusable part.

Every figure carries its conditions. Cold and warm decode differ by ~25% here, a prefill sweep
evicts the decode working set, and a percentage quoted off a curve without its context length means
nothing — so figures without conditions attached are not quotable, including our own earlier ones.

---

## Upstream

This fork tracks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Upstream documentation
applies for everything not listed above; see [the upstream README](https://github.com/ggml-org/llama.cpp#readme)
for supported backends, model conversion and the general tool set.

Bugs found here that belong upstream are noted as such in the model logs.
