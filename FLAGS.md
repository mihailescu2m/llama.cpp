# Fork flags

Flags this fork adds on top of upstream llama.cpp: environment variables (checked once, at startup)
and command-line options. Impact figures come from the A/Bs in `docs/development/roadmap.md`,
`UBATCH.md`, the model notes in `docs/` and the commit messages. They were measured on an M1 Max 64 GB with
Qwen3.8-Flash-Next or DeepSeek V4 Flash. "Not measured" means no A/B isolated the flag.

Unless a row says otherwise, every flag leaves the output unchanged; it only moves time or memory.

## MoE expert streaming

| Flag | Default | Description | Impact |
| --- | --- | --- | --- |
| `--moe-stream` | off | Stream routed expert weights from the GGUF on SSD into an expert cache instead of loading them. | Required to run the 107 GB models on 64 GB at all. |
| `--moe-stream-cache <NG\|Ns>` | auto | Expert cache size: GiB, or slots per layer with an `s` suffix. | The largest single lever: 20 -> 28 GiB halved misses (+30% decode), 28 -> 34 GiB +10%. |
| `--moe-stream-io-threads N` | auto | I/O threads for expert reads. | Not measured. |
| `--moe-stream-direct` | off | Read experts with O_DIRECT (page cache bypassed); falls back to buffered reads. | Not measured. |
| `--ubatch-size-decode N` | 0 (off) | Phase switching: after prompt processing use a small decode ubatch and give the freed workspace to the expert cache (single-slot server). | Qwen decode +10-12% (cache 281 -> 354 slots); DeepSeek decode +14%. |
| `--moe-stream-cache-decode <auto\|NG\|Ns\|0>` | auto | Decode-phase cache size: auto uses the reclaimed workspace, 0 keeps the cache fixed. | Part of the phase-switching gain above. |
| `--prompt-decode-max N` | 400 | With phase switching, uncached prompt tails up to N tokens run at the decode ubatch instead of a round trip. | Measured crossover: staying in decode wins by 166% at 100 tokens and 6.6% at 350, and loses by 14% at 500 and 19% at 600. |
| `LLAMA_MOE_STREAM_LOOKAHEAD` | n_expert_used | Prefetch width: at layer L, predict layer L+1's experts from its router and start those reads early. 0 disables. | DeepSeek decode 6.0 -> 7.3 t/s at width 6; Qwen width 16 -3.2% / -2.5% ms/step at 16k / 64k (the serve script sets 16). |
| `LLAMA_MOE_STREAM_PARTITION` | on | Give each (token, expert) pair to exactly one prefill wave. | Prefill +37%. |
| `LLAMA_MOE_STREAM_WAVE_CAP` | planner | Force the experts per wave (A/B of the wave count). | Tuning only. |
| `LLAMA_MOE_STREAM_PAIR_SLACK` | 50 | Percent slack on the partition chunk over the mean wave's pairs. | Tuning only. |
| `LLAMA_MOE_WAVE_SLACK` | on | Size waves for one fewer slot than the cap. | Not measured. |
| `LLAMA_MOE_STREAM_PAD_SKIP` | on (Metal) | Drop partition-path padding pairs from the MUL_MAT_ID work. | Prefill-heavy session -3.7% prompt time. |
| `LLAMA_MOE_STREAM_NO_PRELOAD` | unset | Disable preloading the next wave's experts during the current wave. | Not measured. |
| `LLAMA_MOE_STREAM_SPEC_MAX` | 8 x I/O threads | Cap on queued speculative (prefetch) reads. | Not measured. |
| `LLAMA_MOE_STREAM_WARM` | on | After a grow to the decode cache, refill empty slots with the experts the previous decode used. | First decode after a grow -6.7% ms/step, misses -44%. |
| `LLAMA_MOE_STREAM_INPLACE` | on (Metal) | Resize the expert cache within its allocation, releasing tail pages, instead of copying. `=1` also resizes CPU buffers (tests only). | Transitions 2.9 -> 1.1 s; session wall -4.4% / -5.5%. |
| `LLAMA_MOE_STREAM_HOT_DECAY` | 1024 tokens | Halving interval of the eviction hotness counters (0 = never decay). | Tuning only. |
| `LLAMA_MOE_STREAM_NO_ZEROCOPY` | unset | Stage reads through a bounce buffer even when the cache is host-visible. | Measured no better than direct reads. |
| `LLAMA_MOE_STREAM_NO_HASH_PREFETCH` | unset | DeepSeek: do not preload the hash-routed layers' experts before layer 0. | Not measured. |
| `LLAMA_MOE_STREAM_STATS_MS` | off | Print stream statistics every N ms. | Diagnostics. |
| `LLAMA_MOE_STREAM_DEBUG` | off | Verbose stream logging. | Diagnostics. |
| `GGML_METAL_RESIDENCY_KEEP_ALIVE_S` | while running | Keep the Metal buffers wired for as long as the process runs (upstream: 180 s after the last compute). `N > 0` releases them after N idle seconds. | Qwen after 6 min idle + 12 GiB pressure: next request 0.17 s; with the 180 s default the model was compressed (56 GiB) and swapped (+9.6 GB) and the next request took ~40 s. |
| `LLAMA_SPEC_DRAFT_MOE_SLOTS` | 96 | Expert cache slots for a streamed draft model (0 keeps the target's budget). | Keeps a small drafter from taking memory the target needs. |

## Profiling

| Flag | Default | Description | Impact |
| --- | --- | --- | --- |
| `GGML_METAL_KPROF=<stride>` | off | Per-kernel GPU time attribution, one sample every `stride` nodes. | Adds ~30 us per node while on. |
| `GGML_METAL_KPROF_DEBUG` | off | Log the KPROF sampling itself. | Diagnostics. |
| `GGML_METAL_GPU_PROFILE` | off | Per-context GPU busy time, printed at exit. | Diagnostics. |
| `LLAMA_BENCH_FIXED_ARMS` / `_ARM` / `_CORPUS` / `_LOGITS_DIR` / `_DECODE_UBATCH` / `_DECODE_CACHE` | unset | Probe arms, frozen corpus, logit dumps, and a decode-phase ubatch/cache (`auto` or slots) to measure at production size. | Measurement harness. |

## DeepSeek V4

| Flag | Default | Description | Impact |
| --- | --- | --- | --- |
| `LLAMA_DSV4_UNION` | on | Union-8 sparse attention: 8 queries share one deduplicated top-k list, with exact per-query membership. 0 selects the per-query sparse path. | Exact. The notes give the design, not an end-to-end number. |
| `LLAMA_DSV4_UNION_MIN_NCSA` | 4096 | Compressed-KV length from which union-8 is used. | Crossover gate. |

## Qwen3.8-Flash-Next

| Flag | Default | Description | Impact |
| --- | --- | --- | --- |
| `LLAMA_QSA_GATHER` | off | Gathered QSA prefill: flash attention reads only the selected cells, 8 queries sharing their union. **Not bit-identical** (different summation order). | Cold prefill -7.4% at 50k tokens, -2.8% at 9.6k; answers identical. |
| `LLAMA_QSA_GATHER_MIN_KV` | 8192 | KV length from which the gathered path is used. | Crossover gate. |
| `GGML_METAL_FA_SPARSE_HR` | on | Sparse FA prefill with the query heads as the matrix rows: one threadgroup per (token, KV head), each selected K/V row loaded once for the 12 heads sharing it. The Metal backend runs dense FA below 3x `top_k` KV (12x with this off, for the vec kernel). `=0` falls back to the vec kernel. **Not bit-identical** to the vec kernel (different summation order). | Prefill vs the best previous path (dense below 24k, vec above): +5% at 8k, +16% at 16k, +18% at 32k, +17% at 64k; kernel 3.5-5x faster than the vec kernel. |
| `LLAMA_PLE_PREFETCH` | on | Advise the kernel of every per-layer-embedding row a ubatch will gather. | Decode -4.8% ms/step, prompt -6.7%. |

## Server and speculative decoding

| Flag | Default | Description | Impact |
| --- | --- | --- | --- |
| `--spec-mtp-ngram-n-max N` | 0 (off) | Append up to N n-gram tokens after a full MTP draft. **Changes greedy text** (verify width changes rounding). | Code edits +8.0% decode, prose -1.7% (N = 3). |
| `--spec-max-prompt N` | 0 (unlimited) | Disable speculation for prompts longer than N tokens. | Avoids a drafter pass over very long prompts. |
| `--slot-persist` | off | Save the most recent slot on exit and restore it on start; with `--context-cache-path` the newest entry is preloaded. | 5994 tokens restored in 0.16 s instead of re-prefilled. |
| `--context-cache-path PATH` | off | Spill idle conversations to SSD and keep them across restarts. | After a restart: 258 tokens processed instead of 4724 re-prefilled. |
| `--context-cache-slots N` | 0 (no limit) | Conversations kept in the context cache, oldest evicted first. | Bounds disk use. |
| `LLAMA_SPEC_REJECTION` | off | Verify draft-model proposals by rejection sampling on sampled requests (greedy requests keep exact match). Output distribution unchanged. | Not measured separately. |
| `LLAMA_SPEC_MTP_REJECTION` | off | Same for the MTP drafter. | Acceptance 0.537 -> 0.590, 13.98 -> 14.89 t/s (+6.5%). |
| `LLAMA_SPEC_DRAFT_UBATCH` | 1024 | Cap the MTP drafter's ubatch below the target's (0 keeps the target's). | Keeps the 512-expert head from running at the prefill ubatch. |
