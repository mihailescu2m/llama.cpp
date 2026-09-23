# Decode performance roadmap (Qwen3.8 Flash-Next / DeepSeek V4 on M1 Max 64 GB)

Status (2026-09-14): flags named below that are missing from [FLAGS.md](../../FLAGS.md) were removed or made
unconditional after these measurements; FLAGS.md lists the current ones.

Branch: `codex/phase-ubatch-expert-cache`. Decision metric: fixed-context, fixed-width ms/step
for decode policy changes; end-to-end wall time per turn for workflow changes (phase switching).
Standing rules: no output-altering optimisations; splice, never convert; docs commit last.

## Where the time goes (measured)

- Decode is bound by expert I/O: stall ~12% of decode wall time, miss rate floors near 2.7-4%
  however hard lookahead prefetches; each miss costs ~0.35 ms (~7.4 GB/s, near NVMe speed).
- Expert-cache capacity is the largest single lever: 20 -> 28 GiB halved misses (+30% decode),
  28 -> 34 GiB added +10% with flat misses (less churn).
- Commit 22c645683 (phase-aware ubatch) frees 9.1 GB of prefill workspace in decode and grows the
  cache 281 -> 354 slots (+26%). Each phase round trip costs ~6 s (grow ~3 s + shrink ~3 s),
  ~200 s per 34-turn session before the small-tail skip.
- Each transition is ~2.9 s: allocating the new 28-36 GiB buffer ~1.1 s (0.9 s on shrink), the
  retained-expert copy ~1.65 s, free ~0.1 s, graph reserve 0.04-0.24 s. The copy is page
  first-touch of the new buffer, not memcpy bandwidth (8 threads did not shorten it). The fix is
  to stop reallocating retained slots (in-place resize, Phase 2).

## A/B 2026-09-12 (fixed workload, greedy, 500 tokens/turn, 13 turns)

A = fixed workspace + MTP, B = phase switching + MTP + suffix, C = phase switching + MTP.
Turns 0-4 are realistic (acceptance 0.5-0.8); later turns degenerate into repetition
(`ignore_eos`), so B is only comparable on turns 0-4.

| Arm | Realistic wall s | Decode tg/s | Total wall s |
| --- | ---: | ---: | ---: |
| A | 209.4 | 16.7 | 473.1 |
| B | 201.7 | 18.5 | 483.8 |
| C (n=2) | 200.6 | 18.4-18.8 | 464.2 |

- Phase switching raises decode ~10-12% (the bigger cache) but transitions eat most of it:
  C is only ~2% faster end to end. Cutting transition cost is the lever.
- Small tails (<=166 tokens) cost the same at ubatch 32 as A pays at 4096. Tails of
  375-480 tokens cost +2.7-3.8 s at ubatch 32, still less than a ~6 s round trip.
- The suffix (B vs C) shows no gain on realistic turns (201.7 vs 200.6 s).

## A/B 2026-09-12 #2: decode warm-up and tail-sized prefill

Natural end of turn (greedy, 400 tokens, 13 turns), arms C W F F W C: C = phase switching + MTP,
W = C + `LLAMA_MOE_STREAM_WARM=1`, F = W + `LLAMA_SERVER_PREFILL_FIT=1`.

- Greedy output is NOT reproducible across runs of the same config (w2-W and w5-W split at turn
  2; turn-0 draft counts differ in every arm). Wall time is therefore not comparable across arms
  (the same config gave 312 s and 258 s). Compare ms per verification step,
  `pred_ms / (pred_n - draft_acc)`, or pairs that happened to produce identical text.
- Cause (probe, 2 turns, per-turn text hashes): two spec-off runs were identical; of two spec-on
  runs one matched them and one split in turn 1 on a near-tie. `qwen-serve.sh` enables rejection
  sampling by default, so the drafter draws its proposals from top-40 at random even at temp 0.
  Drafts vary, so verify widths vary, and the target's logits shift in the last bits with batch
  width. Not a correctness bug: output stays the target's argmax up to that rounding.
  Ruled out on the way: radix TOP_K (fix compiled in), the wave planner and partition path,
  Metal atomics (all integer or bitmap).
- Benchmarks at temp 0 now set `REJECT=0` (argmax drafts): same text across arms, exact pairs.
- Bug: for a greedy request the server already verifies by exact match
  (`rejection_verifier_supported()` is false at temp 0), but the MTP drafter reads the
  process-wide `LLAMA_SPEC_MTP_REJECTION` and still proposes its random top-40 draw, with no
  `p_min` cut. Random proposals checked by exact match only lose acceptance.
  Turns 0-5, temp 0: REJECT=1 acceptance 0.588, 18.14 t/s, 2.74 tokens/step, 151.2 ms/step;
  REJECT=0 0.683, 19.83 t/s, 3.02 tokens/step, 152.5 ms/step (w2-W,w3-F,w4-F,w5-W,w6-C vs x1-C).
  Fix: per-request `propose_sampled` in `common_speculative_draft_params` (default true), set
  from `rejection_verifier_supported()`; draft loops propose the mode and apply `p_min` when it is
  false. Output unchanged (exact-match verification either way). Only greedy clients benefit.
- Tail-sized prefill never engaged: every tail was <= 640 (kept in decode) or 2461-2638 tokens
  (above its 2048 limit). The F arms are W replicates. Needs tails in 641-2048, or the 2.5K tails
  run as 2048 + remainder.
- ab5 (tails of 1427-1958 tokens, fit off vs on x3, `REJECT=0`): fit engages (ubatch 2048,
  cache 319 slots instead of 4096/281). Exact-text groups: f3-N vs f2-Y/f4-Y turns 1-8 wall
  188.0 vs 188.0/185.0 s, decode 154.0 vs 152.3/151.2 ms/step; f5-N vs f6-Y 245.7 vs 243.9 s.
  Decode after a fit tail is up to 5-9% cheaper (more experts survive prefill), but ubatch 2048
  adds ~0.6 s per tail and the one-time measurement ~2.5 s. Net 0-1.6% on tail turns: kept
  opt-in. Transitions still ~2.6-3.0 s (still reallocating); revisit with the in-place resize.
- Caveat: identical-config greedy runs can still diverge late (turn 7, ~19K context) through
  prompt-cache reuse: the LCP match differed (0.899 vs 0.901), 41 more tokens were re-evaluated
  in a batch, and a near-tie flipped. Compare turn by turn up to the divergence.
- Warm-up, exact pair w4-F/w6-C (identical text on all 13 turns): no difference on turns without a
  preceding grow (within +-1.7%), control slower on the first turn after each grow: +6.0% (turn
  0), +6.4% (turn 5), +3.8% (turn 10); 4.3 s of 184 s decode saved. BUT the first grow queues
  nothing (no decode history yet), so the turn-0 gap cannot be the warm-up.
- Confirmed by ab3 (C W C W C W, `REJECT=0`, stream stats, turns 0-5): all six arms produced
  identical text on every turn, so each C/W pair is exact. First decode after the warm grow
  (turn 5): C 161.6 / 162.1 / 163.2 ms/step, W 151.0 / 151.4 / 151.4 (-6.7%); demand misses
  3291-3366 -> 1844-1907 (-44%); stall 1.64 -> 0.91 s; ~170 hits per turn land on a warm read
  still in flight (promoted). Turn 0 (first grow, nothing to warm) and turns 1-4 unchanged.
  Warm-up is now the default (`LLAMA_MOE_STREAM_WARM=0` disables).
- The first control arm (w1-C, started 19 s after a rebuild) ran 10% slower on every turn; excluded.

| Arm | ms/step all | turn 0 | turns 1-4 | turns 5, 6, 10, 11 |
| --- | ---: | ---: | ---: | ---: |
| w1-C (excluded) | 164.9 | 178.8 | 164.0 | 164.5 |
| w2-W | 147.9 | 167.2 | 142.4 | 149.4 |
| w3-F | 148.5 | 168.9 | 142.9 | 150.1 |
| w4-F | 149.0 | 170.4 | 143.4 | 149.9 |
| w5-W | 150.1 | 171.5 | 144.2 | 150.3 |
| w6-C | 152.1 | 180.7 | 144.4 | 155.2 |

## Phase 0: stabilise the branch (in progress)

| Item | State |
| --- | --- |
| Small prompt tails stay in decode (`--prompt-decode-max`, default 640: break-even ~660 tokens) | works: 8-9 turns/arm kept in decode |
| Parallel retained-expert copy on resize (`LLAMA_MOE_STREAM_RESIZE_THREADS`) | no measurable effect on transition time |
| Reuse the measured auto budget after the first transition | works (2-4 reuses/arm) |
| Instrument the transition (prepare / measure / resize: alloc, copy, free / reserve) | done |
| Transition timing line logged at WARN | built |
| Print n-gram suffix counters at release | written |
| Harness `LLAMA_BENCH_FIXED_DECODE_CACHE=auto\|N` (measure at production cache size) | written |
| Metal radix TOP_K tie fix (index-order ties) + `test-top-k-determinism` | fix in tree; test fails 0/16 without it, passes with it |
| `qwen-serve.sh`: `PHASE`, `NGRAM`, `PDMAX` toggles; suffix only with a drafter | done |
| UBATCH.md status update | after the A/B |
| Commit as reviewable pieces, docs last | needs approval |

## Phase 1: measure and set defaults

- MTP + n-gram suffix, 2026-09-12 (greedy, 4 arms per workload, identical text within a setting):
  explanation turns (drive7) off 23.26/23.83 tok/s vs on 23.09/23.20 (-1.7%, suffix lands 56%);
  code-edit turns (drive_edit) off 23.25/22.85 vs on 24.94/24.85 (+8.0%, lands 85%). Suffix depth
  1/2/3 on code edits: 23.88 / 24.74 / 24.90 tok/s; on explanations every depth is within 1.5%.
  Keep NGRAM=3. Changing the suffix length changes greedy text (a wider verify batch rounds
  differently).
- Suffix gates, rejected. A per-step trace (not kept) shows a suffix is proposed almost only when all
  3 MTP tokens land (96%), and neither the drafter's confidence nor an EMA of MTP acceptance predicts
  whether the suffix lands. A gate on the suffix's own hit rate (EMA < 0.65 -> 1 token) measured
  -0.4% (drive7) and -0.8% (drive_edit). A 7-token verify step costs ~70 ms (prose) to ~140 ms (code)
  more than a 4-token one, but step cost follows the text more than the width, so replaying a trace
  under another depth mispredicts; only server A/Bs decide.
- MTP head projections at Q8_0 (idea from halogen-flash-server, which measured 51 -> 59% acceptance
  for an 8-bit head). Our sidecar was Q4_0 throughout. The 17 projection tensors were spliced byte for
  byte from unsloth's mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf (its nextn.eh_proj split by rows into our
  fc_embd / fc_hidden halves - Q8_0 blocks do not straddle the 2560 boundary - and nextn.hc_head_* taken
  as our nextn.hc_*); experts and output head unchanged. Checked by dequantizing: cos 0.991-0.998 against
  our Q4_0 copies, eh_proj halves 0.996 / 0.998 against fc_embd / fc_hidden. +41 MiB. A/B, N O O N,
  greedy: MTP acceptance 0.700 -> 0.715 (drive7) and 0.904 -> 0.928 (drive_edit); decode 22.58 / 23.04 ->
  23.40 / 23.61 tok/s (+2.4%) and 24.72 / 24.76 -> 25.40 / 25.37 (+2.6%). Text differs between heads
  where acceptance differs (verify width changes rounding) and is identical within each head.
- A/B the two priority features (opt-in until measured):
  - Tail-sized prefill (`LLAMA_SERVER_PREFILL_FIT=1`): tails up to 2048 tokens reserve only the
    ubatch they need (1024/2048, measured once) and keep a larger cache during prefill.
  - Decode warm-up (`LLAMA_MOE_STREAM_WARM=1`): after a grow, refill empty slots with the
    experts the previous decode phase used, at speculative priority, never evicting.
- `mul_mat_id` achieved bandwidth on Metal (JigSawPT reached 72% on CUDA): sizes Phase 3.
- Re-run fixed 10 vs fixed 16 vs adaptive lookahead at the 354-slot decode cache, 16K/64K/~124K,
  resident MTP, with the stable TOP_K. If fixed 16 still wins, make it default and drop adaptive.
  16K done (arms base k16 adaptive adaptive k16 base, 96 steps, width 4): logits bit-identical
  across all six processes (TOP_K fix holds). First pass base 158.2/158.5 ms, adaptive
  155.9/155.9, k16 153.1/153.6 (-3.2%; misses 2798 -> 2412, stall 1.39 -> 1.22 s, +20% bytes
  read). Replay 125.9-129.1 for all arms, within run-to-run spread.
  64K (base k16 k16 base): first pass base 178.7/180.6, k16 174.1/176.2 (-2.5%; misses -14%,
  stall -14%, +18% bytes read); replay 140.9-142.6 for both; logits bit-identical. Fixed 16 wins
  at both depths: `qwen-serve.sh` default is now 16 (backup `qwen-serve.sh.bak-lookahead`).
  Adaptive code not yet removed; ~124K not run.
- Tune `--prompt-decode-max` from real session logs.
  With in-place transitions (~2.2 s round trip) the break-even looked like ~210 tokens. Measured
  (tails of 250-726 tokens, frozen prompts, 640 vs 200 x2): 200 is worse, wall 208.0 / 208.6 vs
  196.7 / 194.9 s (+6.4%). The tails cost the same either way (45.9 vs 45.4 s of prompt time):
  prefill is no faster per token at these sizes, and each extra round trip adds its transition
  and evicts decode-hot experts. 640 stays; a higher limit (e.g. 1024) is untested.

- Drafter sampling mirrors the request (measured: no gain, now opt-in `LLAMA_SPEC_DRAFT_MIRROR=1`): the MTP drafter samples top-40 at temp 1.0 with
  a random seed, whatever the request uses. Under rejection sampling acceptance is sum min(p, q),
  so q should come from the request's own stateless chain in the same order (top-k, top-p, min-p,
  typical-p, top-n-sigma, temperature last); skip penalties, DRY and XTC (history-dependent or
  random), seed from the request. Output distribution unchanged. Server defaults (temp 1.0,
  top-k 40, top-p 0.95, min-p 0.01) already match on temp and top-k; top-p and min-p do not.
  Measure at those settings and at temp 0.6; later try a drafter temp slightly below the request's.
  Implementation: `common_speculative_set_sampling()` at slot launch rebuilds the draft-simple and
  MTP samplers from the request's stateless chain (`LLAMA_SPEC_DRAFT_MIRROR=0` keeps the fixed
  top-40), seeded `seed ^ 0x85ebca6b` - a third stream beside the target's and the verifier's.
  The GPU-side top-40 pre-cut stays; the CPU chain runs over those candidates.
  ab6 (turns 0-5, request seed 42, 2 arms each): temp 1.0 / top-p 0.95 / min-p 0.01 acceptance
  0.553 mirrored vs 0.668 / 0.623 fixed (lower on turns 0-4, turn 0 included); temp 0.6: 0.641
  vs 0.636 / 0.668. The truncated drafter proposes too narrowly; matching the settings does not
  help this MTP head. Kept: the drafter's fixed chain is reseeded from each request by default,
  so sampled runs are reproducible (both mirrored arms were identical; unseeded ones diverge).

## Phase 2: extend

- Batched n-gram table (PLE) row prefetch: hash every row a step needs, one advisory call over
  the whole list, include draft-token rows (JigSawPT: 11.4x on n-gram cost per token).
  Measured: the table is not streamed by moe-stream (register_ple() was never wired up); it is the
  upstream TENSOR_READ_LAZY mmap (POSIX_MADV_RANDOM), so each row is a random page fault inside the
  CPU get_rows, taken ~8 at a time by the compute threads. Decode gathers: 698 calls of ~113 rows,
  8.6 ms each = 9.7 ms/step, 7.0% of 138 ms/step (expert-read stall in the same run: 8.1%);
  prefill gathers 8.2 s of 101 s prompt time. With every row page of a ubatch advised
  POSIX_MADV_WILLNEED in llm_graph_input_ple::set_input (sorted, deduplicated, neighbouring pages
  merged): 0.15 ms per decode gather, decode 137.9 -> 131.4 ms/step, prefill gathers 2.0 s and
  prompt 101.3 -> 93.0 s, identical text (single runs). About 3 ms/step of the 9.5 saved is still
  spent elsewhere - likely the advise call itself or pages still arriving; issuing it earlier
  (after drafting) may recover part. Default on (`LLAMA_PLE_PREFETCH=0` disables). A/B (B A A B,
  identical text in all arms): on 131.79 / 131.32 ms/step, prompt 93.1 / 95.0 s, session 174.7 /
  176.4 s; off 138.16 / 138.13 ms/step, prompt 100.8 / 100.7 s, session 186.4 / 186.4 s - decode
  -4.8%, prompt -6.7%, session wall -5.9%.
- Persistent SSD context cache (merged with --slot-persist): the context cache directory was wiped at
  every start and on exit, so after a restart only the one --slot-persist slot survived. Now one store
  in the slot-persistence file format (tokens in the state file, .ckpt, .hbnd, .drft, .gen commit record
  written last): the active conversation is saved on exit, every entry is indexed again on start from
  the file headers (torn or foreign entries deleted), a hit keeps its file, --context-cache-slots is
  enforced (evictions delete their files), and --slot-persist preloads the newest entry. Without a
  context cache path the old state.bin behaviour is unchanged. End to end (two conversations, spill,
  graceful restart, --prompt-decode-max 0 so batching matches the reference): all 6 turns identical;
  after the restart the other conversation loads from SSD (258 tokens processed instead of 4724
  re-prefilled) and the last one is preloaded (5994 tokens in 0.16 s). Crash (kill -9, torn entry
  planted): the torn entry is dropped, completed entries are indexed, the spilled conversation is
  identical, the active one resumes from its last saved prefix. With --prompt-decode-max 640 a tail
  processed at a different ubatch after a restart (prefill vs decode phase) can change text through
  rounding - batching, not the restore (identical at 0).
  Fixed alongside: test-moe-stream-priority failed ~3% of runs (3/120 at the previous head, 9/300 with
  labelled waits, all at the second gate wait of test_ple_and_shutdown). The worker's gate episode
  only ended when its wait returned, so a worker that saw the speculative queue empty and slept on,
  then was held again, counted nothing and billed the idle stretch as gate time. The episode now ends
  as soon as the worker is no longer held: 0/500.
- Transition shaving (#3): per round trip in a 9-turn session the shrink is ~1.58 s and the grow
  ~0.59 s, plus a one-time ~2.97 s first decode entry (allocate the split cache, copy the residents).
  Re-wrapping the 48 layers' GPU views in parallel: grow 590 -> 424 ms; the shrink is unchanged at
  ~1.58 s - its re-wrap got faster (374 -> 239 ms) but the wait for the driver to unwire the tails
  grew by the same amount, so the shrink is bound by the driver (~1 s after the views drop). Kept.
  Startup round trip while the cache is empty (LLAMA_SERVER_PHASE_WARMUP=1): first request's decode
  entry 2966 -> 397 ms, first turn 65.3 -> 63.1 s, but startup +2.78 s - it moves the cost rather than
  saving it (the residency set wires the untouched tail pages, so the startup shrink still waits
  ~0.7 s), so it is opt-in. Sizing the unwire wait by resident tail bytes (mincore) changed nothing
  and was dropped.
- DeepSeek phase switching (2026-09-13). Allocation audit: the DSV4 raw-SWA and compressed caches take
  n_ubatch once, at creation, from the prefill ubatch; switching changes only the compute ubatch, so the
  prefill capacity is kept. The rest of the branch needs no model code for V4.0: routing already uses
  ggml_top_k, V4's hyper-connections have their own fused kernels, lookahead is generic in build_moe_ffn,
  and the Metal, streaming, padding-skip, compact-dispatch and context-cache changes apply as they are
  (MTP work deliberately not ported). `ds4-serve.sh` gained `PHASE` / `PDMAX`, default PHASE=0.
  Measured on drive7 (8k opening + 8 tails, 200-token answers, i.e. post-prefill decode): PHASE=0 5.22 /
  5.23 tok/s, prompt 185 s; PHASE=1 PDMAX=0 5.95 tok/s (+14%), prompt 204 s (~2 s of transitions per turn:
  shrink 1.5 s, grow 0.45 s), expert cache 97 -> 106 slots in decode, text 9/9 identical to PHASE=0. One
  run only, and it saw +1.26 GiB swap (a 2 GiB file read ran beside it; unconfirmed). PDMAX > 0 does not
  carry over: a 32-token ubatch on V4.0 still touches most experts, so 144-947 token tails took 18-77 s in
  decode against 10-18 s in prefill (prompt 427 vs 185 s). Keep PDMAX=0 on DeepSeek.
- Server fix, both models: n_ubatch was read once per update_slots pass, in the decode phase, so after a
  prompt tail switched back to prefill the checkpoint offsets and near_prompt_end still used the decode
  ubatch and split the last 32 tokens off every tail. That changed greedy text against PHASE=0 (V4.0: 2 of
  9 turns identical) and cost ~5 s a turn on DeepSeek (an extra expert sweep for 32 tokens). n_ubatch is
  now re-read after the switch: 9/9 identical, one prompt batch per tail. Qwen, PHASE=0 vs PHASE=1 PDMAX=0:
  text 9/9 identical, decode 20.08 -> 21.74 tok/s, prompt 65.2 -> 83.8 s (the transitions; Qwen keeps
  PDMAX=640 so short tails avoid them).
- Qwen indexer KV cache without V (upstream #28330, seen in pwilkin's strix-halo work). The indexer
  reads only K, but its cache allocated V too: 12 layers x 131072 cells x 256 f16 = 768 MiB never used.
  The cache is now MLA-shaped, so llama_kv_cache allocates no V (K stays 128 x 1 head); every V path
  already skips a missing V. Same run as before the change: text 9/9 identical. Old context-cache entries
  still load: the indexer section is the last in the state, so its unread V is a trailing suffix; a
  server started on an old-format cache reused 3,887 tokens of it and answered identically to a cold
  start on both turns. The expert cache is sized in GiB (CACHE), so the RAM is headroom until CACHE is
  raised (~0.75 GiB, about 7 slots).
  DeepSeek decode halves after prefill today, so it is the most I/O-bound target.

## Phase 3: GPU

- Union-grouped decode MoE kernel (Cherenkov): dot each weight block against all verify rows,
  reading each expert once per step. Ours re-reads per (row, expert): 4 rows cost 326 us per
  node vs 91 us for 1 row; MUL_MAT_ID is ~26% of GPU time at K=4. Bit-identical per output.
  In progress. Measured inputs: a 4-token verify step touches 29.0-29.2 unique experts of 40
  pairs per layer (0.72), so grouping removes ~27% of expert mat-vec work. Per-node microbenchmark
  (one node, ~228 us submit overhead subtracted): down MXFP4 at 4 tokens ~446 us all-distinct vs
  ~139 us when all tokens share their experts - bound by weight reads, so sharing reads pays;
  gate/up IQ3_XXS ~218 vs ~249 us - bound by decode, sharing reads does not help it.
  v0 (written, since removed): a map kernel groups the step's pairs by expert,
  the grouped kernel runs each group's members back to back in one threadgroup through the
  unchanged per-row kernels (bit-identical by construction), so members after the first read the
  expert from cache. Covers every type. Expected ~6% of decode from down; IQ3_XXS needs a v1 that
  decodes each weight block once for all members (~4% more).
  Decode profile (GGML_METAL_KPROF=1, 4-token verify graphs; the profiler adds ~30 us per node):
  MUL_MAT 46% of GPU time, MUL_MAT_ID 24%, flash attention 6%. MUL_MAT_ID per node: gate/up
  (IQ3_XXS) 306 us x2 per layer, down (MXFP4) 345 us - gate/up is 64% of expert time. So grouping
  saves at most ~2% (down, v0) + ~4% (gate/up, v1) of decode.
  Larger: small dense matmuls are latency-bound. Per node at 4 tokens: hyper-connection inject
  10240->4 (F32, 160 KB) 114.5 us x97/step, hc down 10240->320 (Q8_0, 3.3 MiB) 106.9 us x98,
  outputs 10240 (attn_qkv + hc up) 220.7 us x134, 2560 outputs 91.4 us x99; the output head
  (644 MiB) 3.6 ms, ~45% of peak. Speeding these up bit-identically rules out split-K (it changes
  the summation order); candidates are kernel choices that keep each output's reduction, fusion,
  and overlap of independent nodes. Investigating.
  Decode profile 2026-09-12 (all kernel changes on, 4-token verify graphs): the largest dense nodes
  are hc_inject 3.1%, attn_gate ("z", 2560->6144) 2.7%, ssm_out 2.6%, output head 2.5%, Qcur 1.8%.
  Q8_0 mat-vec microbenchmark (distinct weights per node): attn_gate 68.5 -> 85.5 us at 1 -> 4
  tokens (244 -> 196 GB/s), ssm_out 63.9 -> 97.0 us (262 -> 172), attn_qkv 111.9 -> 138.7 us (249 ->
  201), output head 1.89 -> 2.77 ms (358 -> 244). One token already reaches 60-90% of peak; the 4-token
  verify step is what loses bandwidth. If 4 tokens cost what 1 does, decode would save ~3%; that needs
  a multi-token Q8_0 kernel that keeps each output's reduction order (the existing nsg / r1ptg / NR1
  settings are already swept). Tried: staging each K step's src1 chunks in threadgroup memory once
  for all rows of the threadgroup (same values, same per-thread order: bitwise identical in 24/24,
  test-backend-ops MUL_MAT 1256/1256). Faster only at 2 tokens (-2 to -9%); at 3, 4 and 7 tokens +4
  to +17% on every shape (attn_gate 4 tokens 90.3 -> 94.5 us, output head 2.78 -> 3.09 ms): the two
  barriers per step cost more than the L1 reads they replace. Dropped.
  v0 measured (48-node graph, submit overhead amortized): bit-identical (48/48 cases) but slower
  in every multi-token case - at 4 tokens IQ3_XXS 261 vs 172 us per node, MXFP4 284 vs 151 us
  (realistic overlap); even with every token on the same experts 207/222 vs 171/150 us. With the
  overhead amortized the per-pair kernel costs the same whatever the overlap (172 us at 4 tokens
  for all patterns): it is bound by per-pair compute and latency, not weight reads, so running a
  group's members back to back only removes parallelism and adds the map pass and barriers.
  (The single-node 3x "shared" speedup was a measurement artifact.) v0 removed; a shared-decode
  v1 is capped near ~3% of decode with the same parallelism loss - deprioritized.
  Dense split (`GGML_METAL_MUL_MV_EXT_SPLIT=1`): mul_mv_ext keeps all src1 rows of a step in one
  thread (r1ptg = 4 at 4 tokens), and each output's sum depends only on nxpsg, not r1ptg - so one
  token per threadgroup is bit-identical (42/42 cases vs the default kernel) and gives the
  few-row matrices 4x the threadgroups and a 4x shorter serial chain. Applied when the default
  grid has fewer than 64 threadgroups. Per node, 48-node chain with serialized dispatch, at
  2/4/8 tokens: hc inject 33/81/76 -> 25/42/39 us, hc down (4 tokens) 77 -> 47 us, ssm alpha
  24 -> 15 us; router 2560->512 (64 threadgroups, not split) unchanged - splitting it made it
  slower (26 -> 39 us), as did hc down at 64+ threadgroups under concurrent dispatch.
  Server A/B (REJECT=0, 9 turns, 619 decode steps per arm, two interleaved rounds): identical
  text in all 8 arms; off 149.6 / 148.3 / 148.1 / 148.1 ms/step, on 152.5 (outlier) / 143.0 /
  143.2 / 143.2 - round 2 (on first) is -3.3% on every turn. Default on Metal
  (`GGML_METAL_MUL_MV_EXT_SPLIT=0` disables).
  Profile with the split on (same method): hc inject 114.5 -> 57.4 us, hc down 106.9 -> 68.9 us
  per node. Largest dense item now: hc up 320->10240 (Q8_0, 3.3 MiB) at 244.6 us x96/step, 11.7%
  of decode GPU - K = 320 is not a multiple of 128, so it misses mul_mv_ext and runs the generic
  Q8_0 mat-vec: one threadgroup per token, and 2 of 4 simdgroups idle (10 blocks per row).
  Multi-token Q8_0 mat-vec (`GGML_METAL_MUL_MV_Q8_NR1=2/4/8`): up to NR1 tokens per threadgroup,
  each weight block loaded once, simdgroups past the last block skipped; every output keeps its
  per-thread sums and reduction, so bit-identical (63/63 cases, incl. K=96 and K=1088). Per node,
  serialized, at 2/4/8 tokens: 76/143/275 -> 50/93/177 us with NR1=4 (8 is no better). Server A/B
  (REJECT=0, 619 decode steps per arm, identical text in all 8 arms): round 1 (A B B A) off
  148.8 / 144.9, on 149.9 (prefill also slow - system noise) / 141.8; round 2 (B A A B, with
  GGML_METAL_GPU_PROFILE) off 144.9 / 144.8, on 142.2 / 141.7 ms/step (-2.0%), target GPU busy time
  128.2 -> 126.1 s over the same 70238 command buffers (-1.6%). Default on Metal
  (`GGML_METAL_MUL_MV_Q8_NR1=0` disables).
  mul_mv_ext simdgroups and src1 rows per threadgroup do not change any output's sum (sweep
  nsg 1/2/4/8 x r1ptg 1/2/4: bit-identical). Serialized, 4 simdgroups cut ssm_out 6144->2560 at 4
  tokens 102 -> 74 us, but with concurrent dispatch the default already reaches 75 us, and globally
  nsg=4 made the router erratic and hc down slower. Limited to K >= 4096, >= 1024 rows, >= 3 tokens
  (ssm_out, wo), two server A/B rounds gave -0.4% (clean arms: GPU 126.04 -> 125.53 s, 141.05 ->
  140.46 ms/step); first dropped as drift, but the clean arms separate completely in GPU time (on
  125.21-125.71 s vs off 125.81-126.34 s), the test the Q8_0 MUL_MAT_ID change below passed, so
  kept: default on Metal (`GGML_METAL_MUL_MV_EXT_WIDE=0` disables). Big dense shapes (attn_qkv,
  attn_q, qkv gate) are flat across the whole sweep.
  Measurement note: the second arm of a 4-arm run was repeatedly slower in prefill as well
  (103-108 s vs ~101 s) though prefill does not use these kernels; prompt time (and prefill GPU
  time) marks such arms as noise.
  MUL_MAT_ID scaling (48-node graph): per-pair time is linear in pairs and independent of expert
  overlap (IQ3_XXS gate/up 58/100/183 us at 1/2/4 tokens, MXFP4 down 51/87/159) - so a shared-decode
  v1 saves at most the unique-expert fraction (~27% of gate/up) and only if decode is the whole
  cost. Outlier: Q8_0 down (5 layers) 145/279/545 us, 3.4x MXFP4 on the same shape - the generic
  Q8_0 kernel with 2 rows per simdgroup (51200 threadgroups at 4 tokens) and a simdgroup idle at
  K = 640. 8 rows per simdgroup and idle simdgroups dropped (`GGML_METAL_MUL_MV_ID_Q8_NR0=8`, only
  when the rows divide ne01): bit-identical (72/72, incl. K=2560 and K=96), 545 -> 327 us at 4
  tokens and 145 -> 93 us at 1, the same with concurrent dispatch; 16 rows is worse. The same knob
  on the dense multi-token Q8_0 kernel (hc up) gains nothing (93.7 -> 93.1 us) - not kept.
  Server A/B (two rounds, B A A B then A B B A, identical text in all 8 arms): every clean "on" arm
  has less GPU busy time than every clean "off" arm (125.47-125.84 s vs 126.28-126.86 s), -0.7%
  (expected ~0.55% from 5 layers); decode ms/step 141.1 vs 141.4, below its noise. Default on
  Metal (`GGML_METAL_MUL_MV_ID_Q8_NR0=0` disables).
  All four together (split, multi-token Q8_0, Q8_0 MUL_MAT_ID rows, long-row simdgroups) off vs on,
  B A A B, identical text in all arms and with every earlier run: on 139.05 / 138.59 ms/step, GPU
  124.92 / 124.74 s; off 148.88 ms/step, GPU 132.43 s (the other off arm, 158.75, had slow prefill
  too - noise). Decode -6.8% ms/step, GPU busy time -5.7%.
  Profile with all four on (GGML_METAL_KPROF=1, 4-token verify graphs, ~30 us profiler overhead per
  node): decode GPU 26.7 -> 24.6 s. Per node: hc up 244.6 -> 156.1 us, ->2560 (ssm_out / wo) 97.8
  -> 77.7 us, MUL_MAT_ID down 358.5 -> 322.5 us. Shares now: MUL_MAT_ID gate/up 16.8%, down 8.4%,
  hc up 8.1%, flash attention 7.0%, K=2560 10240-row matmuls 4.1%, ->2560 4.1%, output head 5.6%
  (1 and 4 tokens), hc down 3.5%, hc inject 2.9%; small elementwise ops (MUL, ADD, CONT, GET_ROWS,
  SCALE, SIGMOID, RMS_NORM) ~16% of samples, mostly profiler overhead on ~300k tiny nodes.
  Multi-token Q8_0 kernel, reduction batched: one cross-simdgroup reduction (2 barriers) for all
  src1 rows instead of one per row (2 barriers each) - the same simd_sum steps on the same shared
  layout, bit-identical (63/63 at NR1 2/4/8). hc up at 2/4/8 tokens: 50/93/177 -> 50/85/159 us
  serialized, 90 -> 80 us at 4 tokens concurrent. Refinement of a default-on kernel, no own A/B.
  IQ3_XXS / MXFP4 MUL_MAT_ID rows per simdgroup (2/4/8) x simdgroups per threadgroup (2/4/8): rows
  are reduced independently, so every setting is bit-identical (72/72; the IQ3_XXS table load is
  guarded for more than 2 simdgroups). At 4 tokens, serialized / concurrent: IQ3_XXS gate/up 8
  rows 188.4 -> 179.8 / 177.1 -> 166.5 us; MXFP4 down 4 simdgroups 158.3 -> 153.3 / 152.3 -> 145.5
  us (1 token 44.9 -> 40.8); 2 rows or 8 simdgroups worse, combinations no better. Rule: IQ3_XXS 8
  rows, MXFP4 4 simdgroups, when the rows divide ne01 (`GGML_METAL_MUL_MV_ID_ROWS=0` disables).
  Server A/B, 8 arms (B A A B A B B A), identical text in all, no noisy arm: on 137.88-138.25
  ms/step, GPU 124.02-124.15 s; off 138.93-139.67 ms/step, GPU 124.49-126.08 s. Every on arm beats
  every off arm on both; decode -0.9%, GPU -0.5 to -0.75%. Default on Metal. With all five kernel
  changes decode is ~138.0 ms/step against 148.9 with all off (-7.3%).
  Profile with all five on: decode GPU 24.41 s (26.7 s before any of them); MUL_MAT 40.1%,
  MUL_MAT_ID 25.9%, flash attention 7.2%. The stride-1 profiler serializes every node, so it barely
  shows the expert row layout (gate/up 323.9 -> 322.0 us per node) that the A/B measured.
- Prefill profile (GGML_METAL_KPROF, 8k-token opening turn, ubatch ~4k, 4 expert waves): MUL_MAT_ID
  46.3% of prefill GPU time, MUL_MAT 21.4%, GATED_DELTA_NET 10.1%, flash attention 6.7%; expert reads
  stall 6.5% of prefill wall time. mul_mm_id microbenchmark (256 experts): a 32-token tile costs ~16
  us whether it holds 20 or 32 tokens (10.6 us at <= 16, the half-tile skip), 6.1-6.6 TFLOP/s. The
  same shape with F16 weights is ~20% faster, Q8_0 ~14%, so IQ3_XXS dequant is about a fifth of a
  tile; a 64-token tile (bit-identical: each output keeps its accumulation order) would save at most
  ~15% of MUL_MAT_ID. Tried: 64-token tiles with 8 simdgroups (each still 32 rows x 16 tokens, the
  first 128 threads decoding the weight tile), bitwise identical in 16/16 cases but slower wherever it
  applies (>= 256 tokens): model shapes +5 to +12%, others up to +54%. Half the threads wait on each
  decode and the larger threadgroup lowers occupancy; that outweighs decoding once per 64 tokens.
  Dropped. GATED_DELTA_NET is a per-token recurrence; chunking it changes the arithmetic.
  Loading token t+1's q/k/v/g/beta during token t's reductions (bit-identical) measured slower: 3887
  tokens, S 128, 48 heads, K 4: 65.0 / 67.8 / 70.7 ms per node before vs 73.7 / 86.3 / 74.6 after.
  Dropped.
  Dense prefill MUL_MAT (21.4%) is mostly near peak: z 2560->6144 15.5 ms at 4092 tokens (8.3 TFLOP/s),
  ssm_out 16.3 ms (7.9), Qcur 12288 30.9 ms (8.3). The exception is hc down 10240->320 at 4.5-5.5
  TFLOP/s (2.7% of prefill): 320 output rows and a 320-step K loop, where split-K would change the sums.
  At most ~1% of prefill left there.
- Partition-path wave padding: a wave's GEMMs run a static chunk (1.5x the mean wave's pairs), padded
  by repeating the wave's last pair, so about a third of each wave GEMM recomputed one expert. On
  Metal the padding now carries slot -1, which the mul_mm_id map0 drops, and scatters to the scratch
  row (`LLAMA_MOE_STREAM_PAD_SKIP=0` restores the repeats; other backends and chunks < 32 keep them).
  Microbenchmark (chunk 14577, 9718 real pairs): gate/up 13.42 -> 11.20 ms (-16.5%), down 15.07 ->
  12.96 ms (-14.0%), real rows bit-identical. Server A/B (identical text in every arm): prefill-
  heavy session (8k opening + one tail) prompt 52.60 / 51.65 s on vs 54.80 / 53.55 s off (-3.7%),
  GPU 42.53 vs 45.02 s; full drive7 prompt 92.3 / 92.6 s vs 94.4 / 94.5 s; decode unchanged (22.72
  vs 22.59 tok/s, clean arms). Default on Metal.
- Compact MUL_MAT_ID dispatch: kernel_mul_mm_id's grid was ceil(n_tokens/32) x row blocks x n_expert,
  and nearly every threadgroup found no tokens and returned. A prefill wave GEMM (chunk 14577, 281
  slots) launched 1.28M (gate/up) and 5.1M (down) threadgroups for ~3.4k / ~13.6k with work, and at
  fixed work its time grew ~2x with the grid. map0 now also writes the list of used (expert, 32-token
  tile) pairs and the GEMM runs over (row blocks, list entries), a tile's row blocks back to back
  (expert-major order re-read the token rows per row block and was up to 40% slower). Each tile
  computes what it did before: bitwise identical in 11/11 cases (IQ3_XXS, MXFP4, Q8_0, Q4_K, F16, F32;
  broadcast, top-k and partition inputs with -1 padding), test-backend-ops MUL_MAT_ID 810/810.
  Microbenchmark: wave gate/up 11.53 -> 10.33 ms (-10.4%), down 14.37 -> 9.82 ms (-31.7%), top-10
  shapes -1 to -5%. Server A/B, prefill-heavy session x8 (C F F C, C F F C; one noisy C arm dropped):
  prefill GPU 41.02 / 41.80 / 41.16 s on vs 44.25 / 44.27 / 44.88 / 44.36 s off (-7.0%), prompt 51.2 vs
  53.0 s average (-3.4%); every clean on arm beats every off arm. Text identical in all 8, and in 3
  full drive7 runs. Default on Metal; `GGML_METAL_MUL_MM_ID_COMPACT=0` restores the full grid.
- GDN prefill conv without concat + transposed copy: `ggml_ssm_conv_hist(hist, x, c)` reads the
  d_conv-1 history columns from the cached state and the new tokens straight from x, replacing
  concat(state, transpose(x)) + ssm_conv (`LLAMA_CONV_HIST=0` restores them). Same per-output sum
  order as ssm_conv on Metal (float4 dot for d_conv 4) and CPU (window gathered, then the sequential
  loop): bitwise identical in 18/18 cases, test-backend-ops SSM_CONV_HIST 16/16. Microbenchmark at
  4092 tokens, 10240 channels: 17.0 -> 6.9 ms per layer. In a cold 9.6k-token prefill (KPROF, H O H
  O): prefill GPU 36.77 / 36.70 s vs 37.68 / 37.75 s (-2.6%); the transposed CONT goes 1504 -> 434 ms
  and concat 77 -> 2 ms, while the new kernel costs 541 ms against ssm_conv's 182 (strided token
  reads; a follow-up could take ~0.35 s more). Server prompt times stay within noise
  (prefill-heavy session, drive7 88.5 vs 88.5 s), decode 23.45 vs 23.42 tok/s, text identical in
  every arm.
- QSA prefill FA regrouped by token: every head of a token reads that token's top_k cells, but
  build_attn_mha's FA tiles hold 8 consecutive tokens of one head. From n_kv 32768
  (`LLAMA_QSA_REGROUP_MIN_KV`, `LLAMA_QSA_REGROUP=0` off) qwen4exp runs FA over q [D, 12, n_tokens,
  2]: rows are the 12 heads of one token on a KV head, the mask row broadcast over them with stride
  0 (ggml accepts mask nb[1] == 0; Metal keeps such masks on the tiled kernel). A fully masked block
  is a no-op for a row, so every row's result is unchanged: bitwise identical in 6/6 cases,
  test-backend-ops FLASH_ATTN_EXT 4841/4841 with 9 broadcast-row cases. Random-mask microbenchmark,
  4096 tokens: 1.22x at 16k keys, 1.04x at 24k, 0.90x at 32k, 0.68x at 50k. Real prompt (cold
  50k-token prefill, R O O R): 289.8 / 289.5 s vs 292.3 / 291.3 s (-0.8%), answer identical in all
  four. Neighbouring tokens select nearly the same blocks, so 8-token tiles already skip almost as
  much as per-token tiles; the random masks overstated the gain.
- Gathered QSA prefill (`LLAMA_QSA_GATHER=1`, default off; not bit-identical): the union-8 path
  from DSV4 with a head-256 kernel and `ggml_flash_attn_union_mask_all` (the causal mask also on the
  union entries). FA at 4096 tokens with 90% selection sharing inside 8-token blocks: 0.41x at 16k
  keys, 0.24x at 32k, 0.20x at 50k, 0.15x at 98k (no sharing: 0.49x at 50k); max |diff| 2.4e-7.
  The gathered kernel runs at ~1.7 TFLOP/s: rows go through threadgroup memory and each 8-token
  tile computes its whole ~3.4k-cell union, of which each token uses ~60%. Cold prefill A/B (G B B G): 50k
  tokens -7.4% (256.6 vs 277.2 s), 9.6k -2.8%; answers identical. Gathered decode
  (`LLAMA_QWEN4EXP_SPARSE_DECODE=1`, the verify-side twin) measured flat: 50k 19.46 vs 19.63 tok/s,
  9.6k 20.34 vs 20.84; FA is 2-3x faster in isolation but not what decode waits on. Sharing each
  gathered K/V row across 4 heads (vec Q = 4) was bit-identical but 1.14-1.97x slower: dropped.
- /metrics carries the expert stream (`llamacpp:moe_stream_*`: hits, misses, cold misses, late
  prefetch hits, stall seconds, SSD read bytes / seconds / slabs; cache bytes and slots; memory phase),
  read under the stream lock while llama_decode runs. Its token counters move only when a prompt or
  a request finishes, so live rates come from /slots (n_decoded per token, n_prompt_tokens_processed
  per ubatch). No cost: prefill 171-181 t/s and decode 20.2-21.3 t/s with --metrics off, on, and on
  with a 0.5 s /slots + 2 s /metrics poller; OpenCode with the llama-monitor plugin matched OpenCode
  --pure with --metrics off (decode 13.68 / 12.89 vs 13.33 / 13.29 t/s).
- OpenCode decodes at ~13-19 t/s where a sampled plain prompt runs 21-22: its request (system prompt
  and 11 tools, 7.9k tokens) accepts ~0.41 of drafts even at temperature 0, against 0.50-0.65 for the
  same request without tools. Not the tool grammar (tool_choice none: 0.39-0.46) and not the binary
  (HEAD vs working tree identical, draft head q8proj 0.816 vs shared 0.749). Open lead: the n-gram
  suffix drafter against tool-heavy context (NGRAM=0 untested).

## Phase 4: research

- Routing-oracle ceiling before more predictor work (JigSawPT: perfect routing +30%, no
  predictor came close; our K=16 lookahead gives +4% first-pass).
  Measured (LLAMA_MOE_STREAM_ORACLE record/replay, measurement patch not committed): each decode
  remap's real per-layer expert set recorded (33552 remaps), then replayed as layer L+1's prefetch
  set in an identical run instead of the K=16 prediction. Identical text. Decode 131.56 -> 126.41
  ms/step (-3.9%); decode-window misses 2.5% -> 0.8% of expert touches; expert-read stall 8.7% ->
  1.8%. So better prediction (runner-up, learned predictor) can recover at most ~3.9% of decode; the
  remaining ~1.8% is reads issued one layer ahead that still arrive late, plus layer 0.
- Adaptive MTP depth within <=3: free for us (K reserved at max); optimum tracks acceptance
  (depth 2 ~+10% on prose, 3-4 on repetitive). Needs a method robust to trajectory divergence.
- In-place resize (spike 2026-09-12, 14 GiB shared buffer, 7 GiB tail, same path as ggml-metal:
  vm_allocate -> NoCopy -> residency set): `MADV_FREE_REUSABLE` on the tail takes ~40 ms and
  drops the footprint by the full 7 GiB; `MADV_FREE_REUSE` + refill ~0.7 s, GPU reads correct.
  Today's per-transition cost for the same bytes: residency 173 ms + first touch ~0.6 s, plus the
  retained-expert copy. Design: allocate the decode-size cache once (at the first grow); later
  shrinks drop the tail experts and release the tail pages, grows reuse them and let loads fault
  them in. Round trip ~5.7 s -> well under 1 s.
  DEAD as designed (spike v3, system page counts): the first GPU use wires the whole buffer
  (wired 2.1 -> 16.5 GB even without a residency set), and `MADV_FREE_REUSABLE` on wired pages
  only moves them out of the footprint ledger: free memory unchanged, tail still resident.
  Re-wrap WORKS (spikes v4/v5, 14 GiB buffer, 7 GiB tail): drop the MTLBuffer, return the tail
  with `mmap(MAP_FIXED)`, wrap the retained head in place. Free memory +7.2 GB (immediately
  without a residency set, at the next residency/GPU event with one), head intact, no copy.
  Shrink: tail release 20-37 ms + head re-wrap 0.3-93 ms; grow: re-wrap 0.5-250 ms (the residency
  set faults the zero tail in). Constraint: a wrapper must never span a released range - with a
  residency set, wrapping faults the hole in (+7 GB); without one, the first GPU use does. So
  views = the complement of the released ranges (page-aligned by construction). Wiring follows
  use: idle buffers unwire within ~2.5 s and re-wire on the next GPU use.
  Design: the cache is allocated at decode size once (first grow); shrink moves hot tail experts
  into free head slots, sets each tensor to the prefill slot count, returns each tensor's tail
  (rounded inward to 16 KiB pages), and re-wraps the complement; grow wraps the full range and the
  warm-up refills. Needs a ggml-metal call that replaces a buffer's views and residency set.
  Implemented, opt-in `LLAMA_MOE_STREAM_INPLACE=1`: `ggml_backend_metal_buffer_set_views()` /
  `_has_views()` (all new views are created before the old ones go, so a failure leaves the
  buffer intact); `moe_stream_resize_layer_inplace()` shares the retention ranking with the
  copying path, moves only retained tail experts, re-wraps and releases; `size_bufs()` subtracts
  released bytes. CPU buffers resize logically (unit test `test_inplace`). ab7: text identity
  against the copying path, transition time, free memory and swap.
  ab7: correct (text identical to the copying arm on all 9 turns) and fast (shrink 0.75 s, grow
  0.6 s vs ~2.9 s each; wall 302.7 vs 319.4 s), BUT it swapped: +2 GB at the first in-place
  shrink, peaks 3.5 and 4.3 GB in the two in-place arms, none in the copying arms. Cause (spikes
  v6/v7, ggml residency style): the driver unwires released memory asynchronously, when a later
  residency set registers after the release, or within ~1 s. `set_views` registered the new set
  before the tails were released, and the 8.7 GB prefill workspace was reserved milliseconds
  later while the 7.4 GB tail was still wired. The copying path frees layer by layer over ~2.6 s.
  Fix: `set_views` now takes the ranges to release and returns them between dropping the old
  views and registering the new ones; the transition then waits (in-place mode only, 1.5 s cap)
  until system wired memory fell by 80% of the released bytes, and before an in-place grow until
  the freed workspace unwired. Wait time is logged. Validation: ab8.
  That fix still swapped (ab8, ab9), and so did waiting on wired or free memory: both were the
  wrong signal. Root cause, from in-server tracing of anonymous memory (headroom run, 16 GiB): the
  kernel does not free part of a memory object while the rest stays mapped and GPU-resident. The
  released tails left the process (resident -7.6 GB) but stayed with ggml's one-object-per-layer
  allocation as anonymous memory (+7.6 GB), which the compressor and then swap absorbed when the
  8.6 GB workspace landed. Reproduced in a server-shaped spike (24 x 600 MiB buffers, 3 tails
  each): every release method orphaned the tails - MAP_FIXED, FREE_REUSABLE + MAP_FIXED,
  vm_deallocate + reallocate, FREE_REUSABLE alone. It also explains why only the first in-place
  shrink of each run swapped: later ones released tails that MAP_FIXED had already made separate
  objects. Allocation flags do not help: the purgeable flag also works per whole object.
  Fix (implemented): the decode-size cache is allocated at the first grow with each tensor's tail
  [prefill slots, decode slots) as its own memory object (`ggml_backend_metal_buffer_type_alloc_split`,
  vm_allocate FIXED|OVERWRITE before anything wraps the buffer); in-place shrinks on Metal are
  allowed only back to that split point, other targets take the copying path. `set_views` now
  registers the new views before dropping the old ones, and the shrink waits (settled wired drop,
  3 s cap) for the driver to unwire the tails before unmapping them.
  Measured, 16 GiB headroom run: anonymous memory back to baseline at the release (+5.1 GB free at
  once), compressor flat, reserve 228 ms. 28 GiB, one full 9-turn arm (5 round trips): swap 1278
  -> 1278 -> 1270 MiB (no growth), shrink ~1.55 s (unwire wait ~0.63 s, reserve ~0.24 s), grow
  ~0.62 s, vs ~2.9 s each on the copying path: ~3.5 s saved per round trip.
  Same-tree comparison (ab11, 28 GiB, drive5 tails, `REJECT=0`): both in-place arms produced the
  copying arm's text on all 9 turns. Wall 297.7 / 294.4 s vs 311.4 s (-4.4% / -5.5%); transitions
  after the first 1.09 / 1.07 s vs 2.86 s; decode 152.2 / 150.6 vs 150.1 ms/step (the first arm's
  +1.4% did not repeat: +0.3%); swap flat in every arm. GPU read bandwidth over a buffer split
  into 49 objects: 354.6 vs 357.2 GB/s (+0.7% time at full bandwidth; decode reads experts at
  15-25% of peak). In-place is now the default on Metal (`LLAMA_MOE_STREAM_INPLACE=0` disables;
  `=1` also resizes CPU buffers, logically, for the unit test).
  Default checked without the env var (28 GiB, 9 turns): every transition after the first in
  place (48 layers), swap 1270 -> 1270 -> 1238 MiB, transitions after the first 1094 ms mean.
  Caveat for future A/Bs: the drivers build their prompts from the live source tree, so runs
  before and after source edits see different prompts; snapshot the files first.
- Residency-set pool over the file mapping (Cherenkov): resize becomes a budget change, no copy.
  Catches: GGUF experts are not page-aligned; page cache serves little expert traffic here.
- Resident/missing overlap (Cherenkov/Astra): only with the original ordered sum.

## Parked

- RAM split expert cache vs page cache: on 64 GB the expert cache wins; revisit on 128 GB.
- Prefill ring: ubatch 4096 is the best prefill setting, so a ring only costs speed.
- Lookahead held until demand completes: tested, no gain.

## Excluded (output-altering)

2-bit miss experts, deadline cuts on weak experts, q8 KV, derived low-bit stores,
resident-first accumulation.

## Lessons that constrain every experiment

- Verification pays for the union of experts across all drafted tokens on a disk-bound box:
  DeepSeek ngram n_max 24 fell below no-speculation; DSpark's +3.8% was cancelled by the 2 GiB
  of cache it displaced. Drafter weights and wider verification both cost expert residency.
- ngram-mod is shared across requests: repeated prompts replay memorised output (fake +33%).
- `ignore_eos` + greedy degenerates into repetition and flatters drafters.
- Acceptance ratio includes suffix proposals; use mean accepted length and ms/step.
- Cross-process logit checks above 2048 QSA blocks need the TOP_K tie fix.

## References

- Astra's I/O notes: `docs/development/moe-io-overlap.md`; phase design: `UBATCH.md`.
- Cherenkov: https://github.com/alfredr/cherenkov (engine.md, experts.metal, residency.rs).
- JigSawPT DeepSeek-V4.1 on RTX 5090: https://github.com/JigSawPT/deepseek-v41-flash-on-5090,
  fork notes `README-dsv41.md` on branch `dsv41-porte`, discussion ggml-org/llama.cpp#28766.
