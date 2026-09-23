# Decode IO scheduling: implementation notes

Status (2026-09-14): flags named below that are missing from [FLAGS.md](../../FLAGS.md) were removed or made
unconditional after these measurements; FLAGS.md lists the current ones.

Scope: Qwen3.8 XXS on Apple Silicon, CPU-managed expert slots, resident MTP,
and fixed-position target verification. Measurements use ms/step at a fixed
context and width; generated tokens/s is not the decision metric.

## Demand completion policy

`LLAMA_MOE_STREAM_PREFETCH_AFTER_DEMAND=1` prevents idle IO workers from starting
new speculative reads while required expert slabs or PLE rows are outstanding.
The default remains `0` (queue priority). Required reads remain parallel.

The manager tracks work already removed from the queues. When an in-flight
prefetch becomes a demand hit, promotion updates its classification without
issuing another read or changing the slot's remaining-slab count. Completion,
read failure, queued promotion, and shutdown wake the appropriate waiters.
Already-issued speculative reads cannot be cancelled by this policy.

This changes scheduling only. It does not add cache slots, change quantization,
skip experts, alter MTP depth, or reorder the model's expert sum.

The policy is **not generally new for the default Qwen lookahead path**:
`llama_moe_stream_remap_la` already enqueues next-layer lookahead after remap's
demand wait when `LLAMA_MOE_STREAM_LOOKAHEAD_EARLY=0`. The additional worker gate
can affect early lookahead, an old speculative backlog, and hash-route prefetch.
If `while-demand` and `gate waits` stay near zero, a speed change should not be
attributed to removal of demand/prefetch contention.

Telemetry:

- `prefetch dispatch`: speculative slabs actually dispatched.
- `while-demand`: dispatched while required expert/PLE reads were outstanding.
- `gate waits`: episodes in which a worker had speculative work but waited for demand.
- `worker-ms`: summed worker gate waits, **not** elapsed step latency or GPU idle time.
- `expert read MiB`: successful payload reads, excluding PLE and alignment overhead.

`test-moe-stream-priority` uses real workers, temporary-file reads, and a CPU
buffer upload held at a controlled boundary. It checks queue-empty/in-flight
demand, both gate settings, repeated in-flight promotion, queued promotion,
prefetch parallelism after release, PLE gating, read errors, stale work, shutdown,
and cache resizing after drainage.

## Fixed-position experiment

Run `scripts/bench-moe-prefetch.py --out <new-directory> --depth 8192` for a
base/gate/gate/base experiment. It preserves the source/binary hashes, a fixed
token corpus, commands, environmental overrides, memory/swap/power snapshots,
per-step times and token hashes, and first-step logits in each process.
Add `--early-lookahead` to exercise the path that queues prefetch before demand
completion; this flag is held constant in both arms.

Each arm prefills independently, then reduces the target workspace to ubatch 32
while keeping its expert cache at 28 GiB. Context capacity is 131072, verification
width is 4, and every timed step rolls back to the same position. An identical
continuation sequence warms every arm before measurement. This is a target-step
probe, not an end-to-end resident-MTP or acceptance measurement. The OS file cache
is not purged. Do not rebuild or edit measured sources while a run is active.

### 2026-09-11 gate results

**Not isolated; do not use these runs to choose a production setting.** A later
process audit found an old idle `llama-server` still listening on port 8080, with
a large paged-out allocation. It exited during the subsequent HC experiment.
Low CPU use and RSS had hidden its memory footprint. The harness now refuses
other llama model processes before each arm and checks again after each arm.
The queue telemetry and checked-logit observations below remain useful, but the
small latency differences are confounded by system memory state.

Two separate base/gate/gate/base experiments, 96 warm-up and 96 measured steps
per run, position 8192 and verification width 4:

| Lookahead enqueue | Base ms/step | Gate ms/step | Point-estimate reduction |
| --- | ---: | ---: | ---: |
| After demand (default) | 127.578 | 125.523 | 1.61% |
| Before demand (experimental early mode) | 127.446 | 126.003 | 1.13% |

Neither is a demonstrated improvement: repeated gate runs differ by about 4 ms,
larger than their difference from base. With default enqueue timing, almost no
speculative work started during demand even without the gate. With early enqueue,
the first base run had 6285 such slab dispatches and both gate runs had zero;
that confirms the policy engages, not that it improves critical-path latency.
Every arm read the same 97310.79 MiB of expert payload across the whole run
(including prefill). All checked logits and rollback results were bit-identical;
source/binary fingerprints were unchanged within each experiment.

Artifacts: `~/Documents/llama/astra-2026-09-11/prefetch-8k-4/` and
`~/Documents/llama/astra-2026-09-11/prefetch-8k-4-early/`. Leave the gate off.
The incomplete `prefetch-8k-4-hc/` run was deliberately stopped after the process
discovery; it is not a completed A/B. `prefetch-8k-4-hc-isolated/` is the replacement.

## Mixer-aware lookahead experiment

`LLAMA_QWEN4EXP_LOOKAHEAD_HC=1` applies the next layer's learned FFN HC mixer to
the current wide residual before predicting that layer's experts. Inspired by
the next-layer `hc_read_b` in Cherenkov's `streaming.rs`, this avoids using the
current layer's differently trained mixer as the next router's input. It still
omits the intervening FFN and attention updates, so better predictions are a
hypothesis, not a guarantee.

Only streamed Qwen4Exp single-wave batches of at most 32 rows use this path.
It does not change the actual router input, probabilities, selected experts,
expert reduction order, MTP state, or cache capacity. Large prefill and native
MTP graph construction are unchanged. Additional mixer work and temporary
activations are real costs; this switch is off by default pending measurements.
Use `--arms base,hc,hc,base` in the same harness. Flags must not change within a
live process/graph; all arms use fresh processes.

The updated probe emits `FIXED_IO` for the measured sequence only: expert read
payload, demand-stall time, CPU remap time, demand misses, ready/late hits, and
speculative slab dispatches. Unlike the final manager counters, this excludes
prefill and warm-up. Asynchronous speculative reads may straddle its boundaries;
it measures completions in the window, not an exact attribution to each token.

### Isolated HC result: leave disabled

The replacement base/hc/hc/base run completed with no competing llama model
processes at arm boundaries, identical checked logits and rollback results, and
unchanged source/binary hashes. Position 8192, width 4, cache 28 GiB, 96 measured
steps per process after replaying the same 96-step continuation once:

| Metric | Base | Next-layer HC |
| --- | ---: | ---: |
| Mean warm ms/step | 127.590 | 137.652 |
| Expert payload MiB/step | 45.548 | 36.031 |
| Demand stall ms/step | 1.565 | 1.500 |
| Demand misses per 96 steps | 768 | 754 |
| Speculative slab reads per 96 steps | 3222 | 2250 |

HC reduces payload by 20.9% but slows this warm workload by 7.9%. Most of the
saved reads are speculative; demand misses barely change. The first-pass
(warm-up) means are 174.705 ms base versus 177.253 ms HC, also no clear win.
No recommendation to enable HC follows from these results.

This warm probe spends only about 1.2% of step time inside demand-read waits.
It is therefore not a representative test of an IO-bound serving session.
Before estimating benefits from resident/missing overlap, measure first-pass
and varied fixed-depth workloads (including deep context and resident MTP)
without warming the entire measured continuation into the expert cache.
Keep fixed context, fixed width and matched token trajectories in every A/B.

## Cheap prefetch selection experiments

`LLAMA_MOE_STREAM_LOOKAHEAD_ROWS` accepts `last` (default), `first`, and
`round-robin`. For batches up to 32 rows, the latter chooses one new candidate
per row in turn from the already-computed next-router logits. The total budget
remains K unique experts across the whole batch, never K per row. Larger batches
retain last-row selection. For unbiased routing, round-robin ranks raw logits
within each row; biased routing retains the existing transformed score.

The actual router, selected experts, and expert arithmetic are untouched. The
new selector also clamps K to the expert count and ignores nonfinite hints.
Tests cover duplicates, budgets, row offsets, biased ranking, invalid scores,
and the prefill fallback. The working set is small CPU scratch, not more MTP
rollback slots or expert-cache reservation.

The harness accepts `rr`, `first`, `off`, and `kN` arms in addition to the earlier
arms. `--corpus` freezes an existing corpus across experiments. It now records
first-pass ms/step and `FIXED_FIRST_IO` before warming the same continuation,
as well as replayed ms/step and `FIXED_IO`. Both phases hold context and width
fixed and validate the same token trajectory. First-pass cache state follows
the identical prefill and two correctness-check batches; it is not an OS-cache
purge or a claim that every expert is cold.

The initial sweep is `base,rr,k4,off,off,k4,rr,base`, position 8192, width 4,
64 first-pass and 64 replay steps per arm, expert cache 28 GiB. Reducing payload
alone is insufficient: assess demand misses, wait time, and full ms/step together.

Completed results in `~/Documents/llama/astra-2026-09-11/prefetch-cheap-8k-4/`:

| Predictor | First-pass ms/step | Warm replay ms/step |
| --- | ---: | ---: |
| Last row, K=10 (base) | 191.847 | 124.634 |
| Round-robin rows, K=10 | 197.721 | 125.724 |
| Last row, K=4 | 201.050 | 127.064 |
| No lookahead | 207.028 | 123.048 |

All eight checked logit dumps and rollback checks agree exactly; source/binary
provenance stayed unchanged. First-pass base is faster than all alternatives.
No-lookahead saves only about 1.3% in warm replay and loses about 7.9% in the
first pass, so this is not a reason to disable prefetch generally. Leave the
default last-row K=10 configuration unchanged pending broader workloads.

`LLAMA_MOE_STREAM_LOOKAHEAD_ROW_ONLY=1` is the next opt-in experiment: for
single-row predictors and decode batches up to 32 rows, evaluate the next router
only on the row it consumes. Round-robin still calculates every row, and larger
prefill is unchanged. Early-lookahead eligibility follows the actual routed
batch width, not this reduced prediction width. Harness arms `row` and `row16`
select this optimization at the usual K and K=16 respectively.

### 2026-09-12: single-row and wider lookahead

Completed `base,row,k16,k16,row,base` in
`~/Documents/llama/astra-2026-09-12/prefetch-row-8k-4-isolated/`. Same corpus as
the previous sweep, position 8192, width 4, 28 GiB cache, 64 first-pass and 64
replay steps per process. This remains a target-verification probe, not a full
resident-MTP or 128K-context serving test.

| Predictor | First-pass ms/step | Warm replay ms/step |
| --- | ---: | ---: |
| Last row, K=10 (base) | 193.547 | 123.270 |
| Compute one row, K=10 | 193.000 | 125.217 |
| Last row, K=16 | 185.698 | 124.815 |

K=16 is 4.1% faster on the first pass. Demand misses fall from 56.344 to 49.328
per step (12.5%), and measured demand waits from 25.825 to 22.496 ms/step.
Read payload rises from 194.588 to 243.447 MiB/step (25.1%). In warm replay it
has no demonstrated gain, while payload rises from 42.770 to 78.771 MiB/step.
It uses the same expert-cache capacity and MTP/recurrent settings, not additional
expert slots. No production default is changed based on this single workload.

The single-row arm has identical demand/read counts to base and no convincing
latency improvement. It also changes Metal dispatch: the ordinary four-vector
F32 multiply uses the specialized 2..8-vector path, while one vector takes the
generic path. That is a possible explanation, not a profiled attribution of the
end-to-end difference. A two-row variant could retain the specialized path, but
has not been implemented/tested here.

All six checked logit dumps agree exactly and all rollback checks pass; the
measured source/binary hashes stayed unchanged. No competing model process was
found at arm boundaries. AC power was unchanged. System swap usage increased
from about 949 MiB to 1338 MiB during the run; warm timing differences are small
relative to between-run variation and should not be treated as established gains
or regressions.

The next prefetch candidate is a bounded per-layer budget: use the wider budget
where misses remain frequent, back off where it mostly produces unused reads.
This is adaptive IO scheduling, not adaptive MTP depth. The experimental design
below accounts for newly issued prefetch reducing observed misses. Held-out/deeper
workloads are still required before any automatic policy is recommended.

## Per-layer feedback and adaptive 10/16 prototype (2026-09-12)

Both new switches default off:

- `LLAMA_MOE_STREAM_LOOKAHEAD_FEEDBACK=1`: snapshot next-layer residency before
  prefetch and record up to 16 ranked candidates, including unissued shadow
  candidates 11..16 at K=10. Observe the next demand before remap reserves or
  promotes anything. Attribute newly issued reads as ready, late, lost/replaced,
  or unused **at this next demand**, not unused forever. Existing residents are
  not counted as newly useful reads. Per-layer counters are emitted by the fixed
  benchmark at each measurement-window boundary.
- `LLAMA_MOE_STREAM_LOOKAHEAD_ADAPTIVE=1`: implies feedback; with configured K=10,
  choose 10 or 16 independently for each layer. Initial scope is last-row hints,
  CPU-owned slots, and actual routed batches <=32 tokens. K=16 can be instrumented
  but stays fixed; other budgets and policies are unchanged.

The policy uses two EMAs (alpha 1/8): demanded experts absent BEFORE prediction,
and absent shadow-tail experts subsequently demanded. Four valid observations
are required before widening. Widen at cold EMA >=0.5 and tail EMA >=0.125;
back off at cold EMA <0.25 or tail EMA <0.0625. These are experimental thresholds,
not a fitted or proven optimum. A successful newly issued prefetch cannot erase
its own pressure label, although a resident retained from an earlier step can
legitimately make subsequent calls warm. The policy does not explicitly optimize
read bytes, evictions, or per-expert size; next-demand usefulness is only a proxy.

Attribution requires the immediately following manager remap call, matching actual
batch width and allocation epoch. Slot generation/expert identity protect credit
from reuse. Stale observations, multi-wave prefill, changed width/epoch, or
overwritten predictions reset the policy history. It affects IO hints only, not
routing, mathematical operations, MTP depth, rollback capacity, or GPU buffers.
Metadata is bounded by expert count plus 16 candidates per tracked layer.

Tests cover shadow hits at low K, successful ready/late prefetch, generation loss,
unused reads, warm backoff, duplicate observations, call/epoch/width mismatch,
prefill exclusion, and actual remap/resize integration. The fixed harness now
archives untracked source in addition to the tracked patch and hashes, and checks
logits at the initial boundary AND the end of first pass and replay. All timing
comparisons must have the same instrumentation enabled in each arm.

Initial fixed-position comparison (`prefetch-adaptive-8k-4`): depth 8192, width 4,
28 GiB expert cache, 64 first-pass plus 64 replay steps per process; arms
base,k16,adaptive,adaptive,k16,base. Same frozen corpus as prior experiments,
feedback enabled in all arms, no resident MTP drafter. Mean per step:

| Policy | First-pass ms | Replay ms | First-pass read MiB | Replay read MiB |
| --- | ---: | ---: | ---: | ---: |
| Fixed 10 | 197.361 | 127.979 | 194.588 | 42.770 |
| Fixed 16 | 185.498 | 129.543 | 243.447 | 78.771 |
| Adaptive 10/16 | 189.603 | 126.756 | 228.257 | 51.484 |

Adaptive's first-pass point estimate is 3.93% faster than fixed 10, with 17.30%
more expert payload read; fixed 16 is still faster on this IO-heavier phase.
First-pass demand stall: 27.951 / 22.590 / 23.806 ms for fixed 10 / fixed 16 /
adaptive. Adaptive uses the high budget on 1807/3008 observed layer calls during
first pass and 119/3008 during replay, identically in both replicas. It saves
6.24% first-pass and 34.64% replay payload versus fixed 16. Warm differences are
not convincing latency wins: baseline replicas alone span 125.475–130.484 ms.

All six arms have zero logit differences at initial, first-pass-end and replay-end
boundaries; rollback checks pass. Source/binary fingerprints match. No competing
model process at arm boundaries; AC unchanged. System swap ranged about
1194–1301 MiB. All 47 predicted layers contributed 64 observations per window,
with no invalid associations or lost reservations. These are payload-read counters,
not guaranteed physical SSD traffic. This is a prototype tradeoff, not evidence
to change serving defaults or claim a 128K resident-MTP speedup.

Artifacts: `/Users/marianmi/Documents/llama/astra-2026-09-12/prefetch-adaptive-8k-4/`.

## Resident/missing overlap: next implementation

Inspected Cherenkov main at `7b8b21ff1ac59637019a2b463aa564cbfb0a6a89`:

- [streaming.rs](https://github.com/alfredr/cherenkov/blob/7b8b21ff1ac59637019a2b463aa564cbfb0a6a89/src/qwen4_exp/gpu/streaming.rs):
  router-ready, resident-ready, and misses-ready events release two compute phases.
- [experts.rs](https://github.com/alfredr/cherenkov/blob/7b8b21ff1ac59637019a2b463aa564cbfb0a6a89/src/qwen4_exp/gpu/experts.rs):
  resident and fetched expert work are dispatched separately.
- [residency.rs](https://github.com/alfredr/cherenkov/blob/7b8b21ff1ac59637019a2b463aa564cbfb0a6a89/src/qwen4_exp/gpu/residency.rs):
  packed records and a shared address-based pool provide the storage backing.

Adaptation for this fork:

1. Keep CPU slot ownership and the existing mixed-quant GGUF slab storage. Do not
   make overlap depend on `GPU_SLOT=3`; phase-aware cache resizing currently
   requires CPU ownership. A packed store or global pool is not a prerequisite.
2. Split remap into reservation/classification and completion. Snapshot which
   requested slots were resident at publication. Protect the full demand union
   before eviction, and retain protection through both compute phases.
3. Publish router slots and a per-token/per-expert activity mask through a shared
   Metal event. Execute resident work while workers read misses. Gate the second
   phase on completion of **every slab** of its requested experts.
4. Reuse `kernel_mul_mv_id`'s existing quant-specific dot products. Its shared
   wrapper is the insertion point for selective pair execution for IQ3_XXS,
   MXFP4 and Q8_0. Masked pairs must not dereference an unread slot. Merely
   multiplying an ordinary GEMM output by zero is unsafe and wastes work.
5. Preserve each result at its original `(token, selected-expert-index)` and use
   the original ordered sum. Resident-first partial sums change floating-point
   reduction order. Selecting/scattering the computed per-expert results before
   the ordinary sum avoids that additional source of drift.
6. Avoid an extra CPU scheduler split for the second wait. The existing Metal
   MoE handshake is a starting point, but needs a distinct CPU-managed protocol
   with separate publication/completion milestones. Measure its overhead before
   assuming the overlap pays for itself.
7. Start with Qwen decode widths below the Metal matrix-matrix threshold; leave
   prefill, unsupported backends, and large verification batches on the existing
   path. Add CPU/reference behavior for any new GGML operation explicitly.

Validation must submit the **whole graph** to exercise the handshake. Cover no
hits, no misses, mixed hits, duplicate expert IDs across token rows, slow and
failed reads, cancelled prefetch, slot eviction, pending graphs, resize in both
directions, and MTP rollback. Compare cold and warm logits, then matched-context
ms/step at shallow and deep context. Check that resident work actually overlaps
reads on the GPU timeline, rather than relying on additive per-op timing.

The performance ceiling depends on the IO wait that can overlap resident work;
this design does not reduce the SSD payload. Extra dispatch/handshake overhead
can outweigh the benefit in a warm cache. Keep an A/B switch until measured.
