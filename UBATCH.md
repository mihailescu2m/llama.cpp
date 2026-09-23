# Phase-Aware Microbatch and Expert-Cache Plan

Status (2026-09-14): flags named below that are missing from [FLAGS.md](FLAGS.md) were removed or made
unconditional after these measurements; FLAGS.md lists the current ones.

## Recommended approach

Build an **opt-in, phase-aware memory manager** for one conversation:

- Prefill uses large workspaces and the existing expert-cache budget.
- Decode uses small workspaces and reallocates the released RAM to expert residency.
- Existing KV, recurrent state, MTP state and cached expert contents survive the transition.

Start with Qwen3.8 XXS and the resident shared MTP sidecar. Validate DeepSeek V4 separately afterward.

**The primary success metric is lower decode ms/step at matched context and verification width—not acceptance-dependent tokens/s.**

## Implementation status (2026-09-10)

Work is on `codex/phase-ubatch-expert-cache`. The server can rebuild target/drafter
compute schedulers at prompt boundaries and migrate retained experts one layer at a time.
The initial implementation kept the expert slot count fixed unless an explicit decode cache
size was supplied; workspace reclamation alone therefore did not increase expert residency.

With `--ubatch-size-decode 32`, the streamed target cache now defaults to `auto`:

```text
1. Measure combined target/drafter prefill compute-buffer allocations.
2. Destroy them and reserve real decode-sized buffers without evaluating tokens.
3. Measure the allocation reduction.
4. Choose the largest cache fitting that reduction, less a replacement-layer buffer
   and a 512 MiB margin. Include backend tensor padding in the calculation.
5. Invalidate both graphs, migrate the cache, and reserve graphs for the new capacity.
6. Verify the final compute/cache allocation still fits the measured budget.
```

Use the existing prefill flags, for example `-b 4096 -ub 4096 --parallel 1
--moe-stream-cache 28`, and add:

```sh
--ubatch-size-decode 32 --moe-stream-cache-decode auto
```

`--moe-stream-cache-decode 0` explicitly selects workspace-only switching for A/B tests.
Numeric GiB (`30`) and slot (`300s`) settings remain explicit allocation requests; unlike
auto they are not limited by the measured savings. The feature is still disabled when
`--ubatch-size-decode` is absent or zero. Automatic budgeting currently requires CPU/Metal
with host-visible expert buffers and CPU-managed slot ownership.

The system-RAM guard checks scheduler buffer types, including CPU/Accelerate and
Apple Silicon Metal (`MTL` in this fork). A previous registry-name check wrongly
rejected `MTL` and aborted requests at the decode boundary; that check is fixed.
Discrete-device workspace allocations remain excluded from automatic budgeting.

The `memory_phase_budget` log reports measured reclamation, migration headroom and old/new
slot counts. New capacity is empty initially and fills through demand/prefetch reads.
It does not eagerly read guessed experts just to bring physical RAM usage back up.

CPU regression tests cover the budgeting boundary, repeated migration, mixed slabs,
hotness/LRU retention, failures partway through migration, queued reads, whole-graph use
after rebinding, and scheduler rebuilding with pending logits and populated KV. Full
Qwen/MTP validation and fixed-context ms/step measurements remain required before claiming
a speed improvement or enabling these flags in the normal serve script.

### Measured 2026-09-12 (Qwen3.8 IQ4_XXS, M1 Max 64 GB)

Auto budget: `reclaimed_mib=9131.77`, cache 281 -> 354 slots (+26%), identical on every entry.

Each transition costs ~2.85 s, almost all of it in the expert-cache resize (per transition):

| Stage | Grow 281 -> 354 | Shrink 354 -> 281 |
| --- | ---: | ---: |
| replacement buffer allocation | 1.06-1.10 s | 0.88-0.90 s |
| retained-expert copy (8 threads) | ~1.60 s | ~1.61 s |
| old buffer release | ~0.09 s | ~0.12 s |
| workspace reservation | 0.04 s | 0.24 s |
| budget measurement | 0.08 s first, then 0 (reused) | 0 |

The copy runs at ~17 GB/s with 8 threads: it is page first-touch on a freshly allocated
28-36 GiB buffer, not memcpy bandwidth. Any design that reallocates the whole cache per resize
sits near this floor; the fix is to stop reallocating the retained slots.

Changes since 2026-09-10:

- Short prompt tails (`--prompt-decode-max`, default 640) stay in decode: no transitions.
  At ubatch 32 a tail costs ~10.6 ms/token extra vs a ~5.7 s round trip, break-even ~660.
- The auto budget is measured once and reused; the post-resize memory check still runs.
- `LLAMA_SERVER_PREFILL_FIT=1`: tails up to 2048 tokens reserve a 1024/2048 workspace and keep
  a larger cache during prefill (measured at the first decode entry). Opt-in; not yet exercised
  by the A/B workload (its tails were all <= 640 or > 2048 tokens).
- Decode warm-up (default on, `LLAMA_MOE_STREAM_WARM=0` disables): after a grow, empty slots are
  refilled from the previous decode phase's hottest experts at speculative priority, never
  evicting. Three exact pairs (identical text): first decode after the grow -6.7% ms/step,
  -44% demand misses, stall 1.64 -> 0.91 s; no effect elsewhere.

End-to-end A/B (fixed 13-turn agentic workload, greedy, realistic turns 0-4, two runs each):
fixed workspace + MTP 209.2 s; phase switching + MTP 200.6 s (-4.1%, decode +10-12%);
phase switching + MTP + n-gram suffix 202.6 s (no gain from the suffix).

Measurement notes (details in `docs/development/roadmap.md`):

- With rejection sampling on (`qwen-serve.sh` default), the MTP drafter proposes random top-40
  draws even for greedy requests, so greedy runs of one configuration diverge on near-ties.
  Timed A/Bs use `REJECT=0`; the server now proposes the mode whenever the verifier will use
  exact match.
- In-place resize by `madvise` does not work: GPU use wires the cache buffer, so released tail
  pages stay resident. The re-wrap design (drop the MTLBuffer, return the tail, re-wrap the
  retained head) is implemented opt-in (`LLAMA_MOE_STREAM_INPLACE=1`): transitions ~0.6-0.75 s
  instead of ~2.9 s, but the first versions swapped: the kernel does not free part of a memory
  object while the rest stays mapped, so released tails stayed as anonymous memory and were
  compressed or swapped when the prefill workspace was allocated. The decode-size cache is now
  allocated with each tensor's tail as its own memory object, and the shrink unmaps the tails
  only after the driver has unwired them. At 28 GiB over 5 round trips: no swap growth, shrink
  ~1.55 s, grow ~0.62 s; against the copying path on the same prompts, identical output on all
  turns and -4.4% / -5.5% wall time. Default on Metal (`LLAMA_MOE_STREAM_INPLACE=0` disables).
- Small dense matmuls at 2-8 tokens (hyper-connection inject/down, ssm alpha/beta) get few
  threadgroups, each carrying every token through the whole row. Splitting tokens across
  threadgroups keeps each output's summation order, so output is bit-identical; decode -3.3%
  ms/step with identical text. Default on Metal (`GGML_METAL_MUL_MV_EXT_SPLIT=0` disables).
- The hyper-connection up matmul (K = 320, not a multiple of 128) runs the generic Q8_0 mat-vec,
  one threadgroup per token. Up to 4 tokens per threadgroup, loading each weight block once, keeps
  each output's sums and reduction: bit-identical, decode -2.0% ms/step, GPU time -1.6%. Default
  on Metal (`GGML_METAL_MUL_MV_Q8_NR1=0` disables).
- Q8_0 expert down (5 layers) ran 3.4x slower than MXFP4 on the same shape: 8 rows per simdgroup
  instead of 2, idle simdgroups dropped, bit-identical, GPU time -0.7% (`GGML_METAL_MUL_MV_ID_Q8_NR0=0`
  disables). 4 simdgroups for long-row small-batch matmuls (ssm_out, wo): bit-identical, GPU time
  -0.4% (`GGML_METAL_MUL_MV_EXT_WIDE=0` disables). Both default on Metal.
- All four kernel changes together, off vs on: decode 148.9 -> 138.8 ms/step (-6.8%), GPU busy
  time -5.7%, identical text.
- Expert gate/up (IQ3_XXS) with 8 rows per simdgroup and expert down (MXFP4) with 4 simdgroups per
  threadgroup: bit-identical, decode -0.9% (8-arm A/B, every on arm faster). Default on Metal
  (`GGML_METAL_MUL_MV_ID_ROWS=0` disables). All five changes: ~138.0 ms/step (-7.3%).
- The PLE table is a lazily mapped file region: each decode step's ~64-row gather took 8.6 ms of
  random page faults (7% of decode, 8% of prompt time). Advising every row page of the ubatch up
  front cuts the gather to 0.15 ms: decode -4.8% ms/step, prompt time -6.7%, session wall -5.9%
  (4-arm A/B), identical text. Default on (`LLAMA_PLE_PREFETCH=0` disables).
- Transitions: the 48 layers' GPU views are re-wrapped in parallel (grow 590 -> 424 ms; the shrink
  stays bound by the driver's unwire, ~1.58 s). `LLAMA_SERVER_PHASE_WARMUP=1` takes the first decode
  entry at startup (first request 2966 -> 397 ms, startup +2.8 s).
- The SSD context cache survives restarts: the active conversation is saved on exit, entries are
  indexed again on start, and `--slot-persist` preloads the newest one. After a restart another
  conversation loads from SSD (258 tokens processed instead of 4724), identical text.

## 1. Initial configuration and scope

| Setting | Prefill | Decode |
|---|---:|---:|
| Target microbatch capacity | 4,096 | 32 |
| MTP drafter microbatch capacity | Existing cap, currently 1,024 by default | 32 |
| Target expert cache | Current 28 GiB baseline | Baseline plus safely reclaimed RAM |
| MTP depth | 3 | 3 |
| Context capacity | 128K | Unchanged |
| Parallel conversations | 1 | 1 |

Important constraints:

- Keep logical `n_batch` separate from physical microbatch capacity. Do not reduce the server’s request-acceptance limit to 32.
- Derive the minimum decode capacity from the configured verification and replay widths. Reject incompatible configurations rather than silently shortening drafts.
- Do not increase recurrent rollback depth merely because decode capacity is 32.
- Preserve current quantization, kernels, sampling, prefetch policy and checkpoint settings during the first experiments.
- Keep the feature disabled by default.
- Initially support the current CPU-managed streaming path. Explicitly reject unsupported GPU-owned slot-management modes.

The **6–8 GiB reclaim estimate remains provisional**. It must not become an assumed allocation budget.

## 2. Measure the actual memory opportunity first

Add a small diagnostic harness and structured allocation reporting.

For both target and drafter, record:

- Compute-buffer allocation by backend.
- Expert-cache bytes and populated slots.
- Persistent KV, indexer, recurrent and rollback allocations.
- Output buffers and speculative-state storage.
- Process physical footprint, compression and swap activity.
- Peak memory during transitions—not just the final footprint.

Compare reservations for target capacities **4,096, 512, 128 and 32**, without changing the actual verification workload. Measure the drafter separately.

The older benchmark’s 8.54 GiB target workspace is a starting observation, not a fresh measurement or proof that all those bytes are reclaimable.

The initial budget calculation is:

```text
extra expert budget
    ≤ measured workspace reduction
      − transition-copy headroom
      − safety margin
```

Do not calculate this from macOS “free RAM” alone, or count a large mapped model file as fully resident memory.

**Gate:** demonstrate that destroying/rebuilding the compute allocator actually returns memory. Merely changing `n_ubatch` or resetting graph metadata is insufficient.

## 3. Implement workspace switching independently

The first functional patch should shrink workspaces **without resizing the expert cache**. That isolates correctness and establishes the real memory saving.

The existing reservation path, `llama_context::sched_reserve()` in [src/llama-context.cpp](src/llama-context.cpp), deliberately reserves prefill-sized buffers. Add a controlled phase-specific reservation path rather than changing a scalar and hoping those allocations shrink.

### Prefill → decode

1. Finish the final target prompt batch.
2. Finish MTP processing of that batch and consume its hidden-state outputs.
3. Synchronize target and drafter.
4. Preserve pending logits, output indices, hidden-state carriers and sampler state.
5. Invalidate reusable graphs that reference the old workspace.
6. Destroy the old compute allocations.
7. Reserve decode-sized workspaces.
8. Verify invariants before allowing sampling or drafting to continue.

### Decode → prefill

1. Finish the current verification/replay operation.
2. Synchronize both contexts.
3. Restore the prefill workspace capacity before processing the next prompt.
4. Preserve the conversation state.

Switch at request/phase boundaries, **not between individual decode steps**.

Do not keep both complete workspace allocations resident; that defeats the purpose. Reservation must not run a dummy decode that advances KV or consumes RNG.

For architectures whose persistent caches depend on maximum microbatch size—particularly DeepSeek’s sliding-window storage—retain sufficient persistent capacity for the configured prefill maximum.

## 4. Make expert-cache resizing genuinely memory-safe

This is the largest implementation component.

The current streaming allocator, `llama_moe_stream::alloc_bufs()` in [src/llama-moe-stream.cpp](src/llama-moe-stream.cpp), groups cache tensors into large buffers. There is no independently removable “extra 6 GiB” allocation today.

Allocating a replacement 34–36 GiB cache while retaining the old 28 GiB cache is unacceptable.

### Preferred first design: per-layer cache buffers

Refactor the opt-in path so each streamed layer owns independently replaceable backing.

Then resize **one layer at a time**:

1. Allocate that layer’s replacement buffer.
2. Copy its retained expert slabs directly from the old allocation.
3. Validate mappings and publish the replacement.
4. Release the old layer buffer.
5. Proceed to the next layer.

This bounds temporary duplication to approximately one layer’s replacement allocation, rather than a second entire cache.

It also preserves the existing contiguous expert-tensor representation and GEMM kernels. A segmented base-plus-extension cache could avoid some copying, but would require a more intrusive graph/kernel design; defer it unless transition measurements justify it.

### Preserve residency

On growth:

- Retain every valid resident expert and its metadata.
- Add empty capacity.
- Fill additional slots through the existing demand/prefetch machinery.
- Do not synchronously read gigabytes of guessed experts before producing the first token.

On shrink:

- Retain the most valuable resident experts using the existing replacement policy.
- Compact them into the smaller allocation.
- Preserve hotness information where applicable.
- Do not clear and refill the entire cache from SSD.

The cost of copying resident weights must be measured explicitly. Avoiding SSD reloads does not make migration free.

## 5. Add an explicit streaming quiescence protocol

GPU synchronization alone is insufficient: `llama_moe_stream::worker_loop()` in [src/llama-moe-stream.cpp](src/llama-moe-stream.cpp) can still be reading or uploading into cache buffers.

Before resizing:

- Prevent new graph submissions and prefetch reservations.
- Finish outstanding graph work.
- Drain demand and PLE work safely.
- Settle or cancel speculative work with consistent slot bookkeeping.
- Wait for all in-flight reads and uploads.
- Only then modify allocations and residency tables.

Add an allocation generation/epoch so stale queued work cannot publish into a resized cache. Validate slot bounds before indexing resized arrays.

After resizing:

- Rebuild any capacity-dependent wave plans.
- Update all relevant tensor references and slot tables.
- Invalidate graphs in every context sharing the affected tensors.
- Resume workers and graph submission only after the whole configuration is coherent.

Use a coordinator owned by the model/context pair, not an unmanaged process-global singleton.

Failure behavior must be explicit:

- Failure before publication: retain the previous valid configuration.
- Failure after partial migration: restore a demonstrably coherent configuration or stop the request/context.
- Never continue with uncertain pointers, partially filled experts or mismatched mappings.

## 6. Combine the two mechanisms under a bounded policy

Once workspace-only switching and cache resizing pass independently:

### Entering decode

1. Complete the state-safe boundary.
2. Release large workspaces.
3. Reserve small workspaces.
4. Expand the expert cache within the measured budget.
5. Resume generation.

### Returning to prefill

1. Quiesce execution and streaming.
2. Shrink expert residency to the prefill budget.
3. Release decode workspaces.
4. Reserve prefill workspaces.
5. Resume prompt processing.

Start with explicit cache ceilings, testing **+2 GiB, then +4 GiB**, and only proceeding further if measurements support it. Do not immediately set a 36 GiB default.

Proposed experimental controls would distinguish:

- Feature off.
- Workspace switching only.
- Full workspace-plus-cache switching.
- Decode microbatch capacity.
- Maximum decode expert-cache budget.

Use existing `-ub` for prefill initially. New option names should clearly identify decode-only behavior.

Also handle:

- Cancellation and disconnects at either boundary.
- Fully cached prompts and short prompt tails.
- Slot save/restore and context-cache restoration.
- Model unloading and shutdown.
- A projector being loaded, with actual multimodal requests separately tested or explicitly excluded initially.

Extend runtime graph identity with allocation generations. Review persistent state-cache keys separately: physical buffer addresses must not become persistent identity, and existing saved conversations must not be silently deleted or accepted under incompatible semantics.

## 7. Correctness validation

### Small and synthetic tests

Cover:

- Repeated grow/shrink cycles.
- Empty, partially populated and full caches.
- Mixed quantization and different expert-slab sizes.
- Concurrent slab completions and stale jobs.
- Allocation failure at every migration stage.
- Mapping consistency and byte-identical retained expert weights.
- Cancellation during transitions.

Add whole-graph tests. The ordinary backend-op comparison mode alone is insufficient for validating multi-node behavior.

### Real-model tests

For Qwen, exercise:

- No speculation.
- MTP depth 3.
- MTP 3 plus the n-gram extension.
- Full acceptance, partial acceptance and rejection at every depth.
- Checkpoint-dependent replay.
- EOS with `ignore_eos` disabled.
- Fresh prompts, reused prefixes and restored conversations.
- Repeated prefill/decode transitions near 128K.

For the initial 4K-prefill implementation, allocation changes should not intentionally change the mathematics. Require:

- Unchanged state across a transition that evaluates no tokens.
- Identical pending outputs and speculative carriers.
- Identical RNG state.
- Matching logits and generation against the control.

Investigate differences rather than accepting “the text still looks fine.” Expensive state hashing belongs in correctness runs, not timed performance runs.

## 8. Performance validation

Separate the causes with these experimental arms:

| Arm | Purpose |
|---|---|
| Existing implementation | Baseline |
| Per-layer allocation, fixed capacity | Detect allocation-layout regressions |
| Same-size workspace rebuild | Measure boundary/rebuild overhead |
| Small decode workspace, unchanged cache | Measure reclaimed RAM and isolated effects |
| Small workspace plus larger cache | Measure the actual I/O benefit |

### Decode measurements

Use identical model files, prefix states, verification widths and token trajectories across arms.

Measure at shallow, medium and deep context, including approximately 124K:

- Mean/median and tail ms/step.
- Demand misses and SSD bytes per step.
- Critical-path I/O stall time.
- Prefetch usefulness and late arrivals.
- Transition latency.
- Physical-memory and swap deltas.

Keep cold and warm behavior separate, and use balanced paired run ordering.

**Important:** repeatedly replaying one tiny fixed-position workload can make nearly every expert resident. Include several representative fixed-depth workloads and reproducible continuations so the test still exposes the I/O bottleneck. Do not interpret summed worker-read durations as wall-clock stall time.

### End-to-end accounting

Include migration and first-token costs:

```text
net time saved =
    decode-step savings
    − transition overhead
    − additional prefill cost
```

Report the answer length needed to break even. A steady-state gain that loses on typical responses is not sufficient.

## 9. Tune larger prefill batches afterward

Only after the phase-switching baseline is stable:

- Compare 4,096, 6,144 and 8,192.
- Keep the drafter’s prefill cap unchanged initially.
- Ensure logical `n_batch` accommodates the candidate.
- Audit checkpoint boundaries so they do not accidentally split larger batches.
- Compare identical 24K-token spans at matched starting depths.
- Include near-128K memory pressure, not just short-context runs.

Run a same-cache comparison where memory permits. If a larger microbatch requires a smaller prefill cache, treat that as a separate combined configuration.

The winner is the lowest total elapsed time with safe memory behavior—not the largest batch size.

## 10. DeepSeek and rollout

After Qwen passes, repeat allocation measurements and correctness tests for DeepSeek V4, initially with its n-gram configuration.

Do not assume the same reclaimable memory:

- Its attention and compressor state differ.
- Sliding-window capacity needs its own audit.
- DSpark, if used, needs a separate drafter transition test.

Deliver the work as separately reviewable commits:

1. Allocation and phase instrumentation.
2. Workspace-switching API and tests.
3. Server/MTP boundary integration.
4. Per-layer cache ownership and quiescence.
5. Residency-preserving resizing.
6. Combined memory-budget policy.
7. Full-model validation and tuning results.

**First milestone:** prove workspace reclamation safely, with the expert cache unchanged.

**Second milestone:** turn that measured saving into fewer SSD misses and lower decode ms/step.

Neither milestone should depend on changing acceptance behavior, increasing rollback depth, or trusting the earlier RAM estimate.
