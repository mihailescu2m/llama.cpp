#pragma once

#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// LLAMA_MOE_STREAM_PARTITION: give each (token, expert) pair to exactly one wave instead of running
// every wave over every pair and masking the rest to zero. Read in both llama-moe-stream.cpp and
// llama-graph.cpp, which must agree: the graph sizes the gathered tensors that the planner fills.
bool moe_stream_partition();

// SSD streaming of MoE routed expert weights
//
// Streamed layers do not materialize their ffn_*_exps tensors; instead each weight gets a
// device-side cache tensor of n_slots expert slabs, filled on demand from the GGUF file by an
// id-remapping custom op that runs on the CPU right after the router top-k. The remap only
// changes which cache slot an expert id resolves to - it never changes which experts the router
// selected, so streaming affects latency, not outputs.
//
// Missing experts are loaded by a pool of I/O threads while the remap op waits; eviction is by
// decaying route hotness with an LRU tiebreak. Reads are buffered by default, or O_DIRECT with
// LLAMA_MOE_STREAM_DIRECT=1 (bypasses the page cache; recommended when the model far exceeds RAM).
//
// note: multiple contexts decoding the same streamed model concurrently are not supported -
// one context can evict slots referenced by the other's in-flight graph.

// Read-latency histogram for expert slab reads: upper bounds in microseconds, plus one bucket for
// everything above the last bound.
//
// The mean read time cannot separate a page-cache hit from an SSD read. A hit is a ~3 MB memcpy
// (~0.2 ms), an SSD read is 1-3 ms, so the distribution is bimodal and the mean sits between the
// modes - 3.19 ms/slab is consistent with anything from 0% to 40% hits. This histogram is what
// sizes the page cache as an L2 tier behind the slot cache, which decides whether it is worth
// protecting from prefill (prefill streams ~94 GB of experts through a page cache of a few GB).
static const int64_t MOE_STREAM_READ_BUCKET_US[] = { 100, 250, 500, 1000, 2000, 4000, 8000 };
static const int     MOE_STREAM_READ_BUCKETS     = 8;

struct llama_moe_stream;

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1, // reserved, load queued or in flight
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

// one streamed weight tensor (gate/up/down or fused gate_up) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr; // cache tensor {ne0, ne1, n_slots}

    uint16_t file_idx  = 0; // GGUF split file index
    size_t   offs      = 0; // file offset of the full exps tensor data
    size_t   nb_expert = 0; // bytes per expert slab
};

struct llama_moe_stream_layer;

// Element layout of a layer's GPU residency buffer. Kept in one place because the Metal kernel
// indexes the same buffer with the same arithmetic and the two must not drift.
//
//   [0,           n_slots)            slot -> expert id, -1 = empty
//   [n_slots,   2*n_slots)            slot -> last-use stamp (compare as a wrapping difference)
//   [2*n_slots, 2*n_slots + n_expert) expert -> slot, -1 = not resident
//   tail                              MOE_SLOT_TAIL_COUNT counters
//   requests                          up to n_slots (expert, slot) pairs the CPU must load
struct llama_moe_slot_state_layout {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    int32_t off_slot_expert() const { return 0; }
    int32_t off_last_use()    const { return n_slots; }
    int32_t off_expert_slot() const { return 2*n_slots; }
    int32_t off_tail()        const { return 2*n_slots + n_expert; }
    int32_t off_requests()    const { return off_tail() + 4; }
    int32_t size()            const { return off_requests() + 2*n_slots; }
};

enum llama_moe_slot_tail {
    MOE_SLOT_TAIL_CLOCK = 0, // monotonic use counter; the GPU owns it at mode 3
    MOE_SLOT_TAIL_NREQ  = 1, // (expert, slot) pairs the kernel emitted this call
    MOE_SLOT_TAIL_NBAD  = 2, // verify mismatches, mode 1 only
    MOE_SLOT_TAIL_SEQ   = 3, // call sequence, so a servicer can spot a missed handshake
};

// Userdata for the remap op when one-layer-ahead prefetch is on. The op keeps its normal job for
// THIS layer and additionally predicts the NEXT layer's routing from this layer's router input,
// issuing those loads a layer early.
//
// The prediction is deliberately the cheap one: topk(W_next . x_L). It skips the attention and FFN
// terms between the two layers, so it is a lower bound - measured 72.1% top-6 overlap, and 46.3% of
// the experts that actually STALL at K=6, for 0.35 wasted reads per layer-token. The exact version
// would need layer L+1's attention output, i.e. running attention twice per layer (5 GB/token), so
// it is not worth having.
struct llama_moe_stream_lookahead {
    llama_moe_stream_layer * sl      = nullptr; // this layer, remapped as usual
    llama_moe_stream_layer * sl_next = nullptr; // layer to prefetch into, null = plain remap
    uint32_t                 top_k   = 0;       // how many predicted experts to fetch
    ggml_tensor *            bias_src = nullptr;// next layer's exp_probs_b, may be null
    bool                     bias_read = false;
    std::vector<float>       bias;              // host copy of bias_src, filled on first use
    std::vector<float>       score;             // scratch [n_expert]
};

struct llama_moe_stream_layer;

// userdata of one wave's custom ops (multi-pass prefill): identifies which pass this is
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl   = nullptr;
    int32_t                  wave = -1;
};

// per-layer streaming state - also the userdata of the id-remapping custom op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;

    std::vector<llama_moe_stream_weight> weights; // 2 (fused gate_up + down) or 3 entries

    // residency state, guarded by mgr->mtx
    std::vector<int32_t>                 slot_expert;   // [n_slots] expert id or -1
    std::vector<uint8_t>                 slot_state;    // [n_slots] llama_moe_stream_slot_state
    std::vector<uint8_t>                 slot_pending;  // [n_slots] slabs still in flight for this slot
    std::vector<uint64_t>                slot_gen;      // [n_slots] reservation generation
    std::vector<int64_t>                 slot_last_use; // [n_slots] LRU stamps
    std::unordered_map<int32_t, int32_t> expert_slot;   // RESIDENT and LOADING entries

    std::vector<uint32_t> route_hotness; // [n_expert] decayed selection counts, for eviction
    std::vector<uint8_t>  seen;          // [n_expert] for cold-miss attribution
    int64_t use_counter = 0;

    // scratch for the remap callback
    std::vector<int32_t> uniq;
    std::vector<uint8_t> touched;
    std::vector<uint8_t> keep;         // [n_slots] slots the current call must not evict
    std::vector<int32_t> demand_slots; // slots the current call waits on

    // wave plan for multi-pass prefill (guarded by mgr->mtx): the touched experts are split into
    // plan_n_waves passes of at most plan_capacity experts each, run one pass at a time
    uint32_t plan_capacity  = 0;  // experts per wave, set at graph build
    uint32_t plan_n_waves   = 0;  // waves of the current call
    int32_t  plan_next_wave = -1; // wave expected to run next (ordering guard)
    std::vector<uint8_t> expert_wave; // [n_expert] wave each touched expert belongs to, 0xff = untouched
    std::vector<int32_t> plan_pool;   // resident slots the masked-out pairs of this wave park on
    std::vector<int32_t> pool_used;   // scratch: pool slots already used in the current token row

    // Pair partitioning (LLAMA_MOE_STREAM_PARTITION=1). The default design runs the expert GEMMs
    // once per wave over EVERY (token, expert) pair and masks the other waves' pairs to zero, so GPU
    // cost scales with wave count - measured at ~8.15 s per wave atop a ~26.9 s fixed cost for a
    // 3798-token prefill, i.e. ~33 s discarded at the default 5 waves.
    //
    // Partitioning by TOKEN is impossible here: a token needs n_expert_used experts and a wave holds
    // only plan_capacity of n_expert, so ~no token's picks fit in one wave. Pairs, however, already
    // belong to exactly one wave via expert_wave[], so the GEMM can run over a dense pair list with
    // the expert dimension collapsed to 1, then scatter back. plan_pair_chunk is fixed at
    // ceil(n_tokens*n_ids / n_waves) because ggml shapes are static; short waves pad.
    uint32_t plan_pair_chunk = 0;                // static per-wave bound, set at graph build
    uint32_t plan_waves_want = 0;                // wave count the graph built, 0 = not partitioned
    std::vector<uint32_t> wave_first, wave_count; // [n_waves] this wave's slice of uniq
    std::vector<std::vector<int32_t>> plan_pair;  // [n_waves][<= plan_pair_chunk] flat idx t*n_ids+k
    std::vector<int32_t>              pair_count; // [n_expert] pairs per expert, for wave balancing

    std::vector<std::unique_ptr<llama_moe_stream_wave>> wave_ud; // stable per-wave op userdata

    // stable userdata for wave w (grows lazily); called at graph build time only
    llama_moe_stream_wave * wave_userdata(int32_t wave, uint32_t capacity);

    // whether the exps tensors passed to build_moe_ffn are this layer's cache tensors
    // (e.g. grovemoe evaluates a second, unstreamed expert group on the same layer index)
    bool matches(const ggml_tensor * gate, const ggml_tensor * up,
                 const ggml_tensor * down, const ggml_tensor * gate_up) const;

    // Device-resident residency state for GPU slot resolution (LLAMA_MOE_STREAM_GPU_SLOT), one I32
    // buffer per streamed layer. At mode 3 the GPU owns every field here and the CPU only services
    // the requests the kernel emits; see llama_moe_slot_state_layout for the element layout.
    ggml_tensor * state      = nullptr;
    int32_t *     state_host = nullptr; // resolved lazily, null until the buffer is real
    bool          state_init = false;   // the buffer starts as garbage - do not read counters yet

    llama_moe_slot_state_layout layout() const { return { (int32_t) n_expert, (int32_t) n_slots }; }

    void publish_state_locked(llama_moe_stream & mgr);

    // mode 3: load whatever the resolve kernel asked for, then let the GPU continue. Runs on
    // Metal's listener queue, NOT on the graph thread.
    void service_requests(llama_moe_stream & mgr);

    // one-layer-ahead prefetch (LLAMA_MOE_STREAM_LOOKAHEAD=K). Set by the model at load time so
    // build_moe_ffn can reach the NEXT layer's router without a signature change.
    ggml_tensor * la_gate_inp = nullptr;   // next layer's ffn_gate_inp
    ggml_tensor * la_bias_src = nullptr;   // next layer's exp_probs_b, may be null
    struct llama_moe_stream_lookahead * la = nullptr;
};

// A layer whose expert choice is a token-id lookup instead of a router matmul (deepseek4 sets
// hash_layer_count = 3). Routing there is known before the graph runs, so the loads can be issued at
// the top of the pass rather than at the layer.
//
// Worth its own path because the miss load is wildly uneven: measured at cache 40, layers 0-2 are 3
// of 43 layers but 36% of steady-state decode misses, and their share GROWS as the cache warms
// (14% -> 36% over 200 tokens). They hash the token id, so they gain no locality - 237 distinct
// experts touched over 200 tokens against 113 slots - while the learned-router layers settle.
struct llama_moe_stream_hash_router {
    llama_moe_stream_layer * sl  = nullptr;
    ggml_tensor *            map = nullptr; // tid2eid {n_expert_used, n_vocab}, I32
    std::vector<int32_t>     rows;          // host copy of map, filled on first use
};

// one queued expert load
struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1;
    int32_t  widx   = -1; // which weight slab of the expert; one work item per slab
    uint64_t gen    = 0;  // stale unless it matches slot_gen[slot]
};

// Qwen3.8's PLE table is much larger than RAM but each ubatch touches only a compact set of rows.
// Unlike an expert tensor it needs no persistent slot cache: raw quantized rows are read directly
// into one compact graph input, then the normal get_rows op dequantizes them.
struct llama_moe_stream_ple {
    bool      registered = false;
    ggml_type type       = GGML_TYPE_COUNT;
    uint16_t  file_idx   = 0;
    size_t    offs       = 0;
    size_t    row_size   = 0;
    int64_t   row_ne     = 0;
    int64_t   n_rows     = 0;
};

struct llama_moe_stream_ple_work {
    int32_t   row = -1;
    uint8_t * dst = nullptr;
};

struct llama_moe_stream {
    uint32_t n_slots      = 0; // expert cache slots per streamed layer
    int32_t  n_io_threads = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers; // [n_layer], null = not streamed

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct);
    ~llama_moe_stream();

    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    std::vector<llama_moe_stream_hash_router> hash_routers;
    uint32_t hash_n_used = 0; // n_expert_used, the row stride of every tid2eid

    // idempotent; called at graph build, which is the first point the tensor pointer is known
    void register_hash_router(int32_t il, ggml_tensor * tid2eid, uint32_t n_expert_used);

    // registers a streamed weight of layer il and returns its cache tensor
    ggml_tensor * create_cache_tensor(
            int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
            uint16_t file_idx, size_t offs);

    // allocate the cache tensor buffers (after all create_cache_tensor calls)
    void alloc_bufs(bool no_alloc);

    // reopen the GGUF files for streaming reads
    void open_files(const std::vector<std::string> & paths);

    // Register and read an on-disk PLE gather table without materializing the full tensor.
    void register_ple(const ggml_tensor * meta, uint16_t file_idx, size_t offs);
    void read_ple_rows(const std::vector<int32_t> & rows, ggml_tensor * dst);

    size_t size_bufs() const;

    void print_stats() const;

    // LLAMA_MOE_STREAM_NO_ZEROCOPY: stage reads through a bounce buffer even when the backend offers
    // a host pointer. Kept as a switch because reading straight into the cache measured no better.
    bool no_zerocopy = false;

    bool use_direct_io = false; // O_DIRECT streaming reads (LLAMA_MOE_STREAM_DIRECT), no page cache

    llama_files files; // privately reopened GGUF files, same indices as the loader's

    llama_moe_stream_ple ple;
    std::unique_ptr<llama_file> ple_file; // always buffered; tiny PLE rows benefit from the page cache

    size_t  max_nb_expert      = 0;
    int64_t hot_decay_interval = 0; // remap calls between route-hotness halvings (0 = no decay)

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> ctxs; // one per buft
    std::vector<ggml_backend_buffer_ptr> bufs;

    // The GPU slot tables get their own context and buffer, so that turning the feature on leaves
    // the expert-cache buffer byte-for-byte as it is with the feature off. Sharing the cache
    // context would interleave a small I32 tensor between a layer's expert slabs.
    ggml_backend_buffer_type_t buft_state = nullptr;
    ggml_context_ptr         ctx_state;
    ggml_backend_buffer_ptr  buf_state;

    // load pool (queue and all layer residency state guarded by mtx)
    mutable std::mutex      mtx;
    std::condition_variable cv_work; // queued work or shutdown
    std::condition_variable cv_done; // a load committed or failed

    std::deque<llama_moe_stream_work> q_demand;

    // PLE rows are graph inputs and must all land before graph execution starts. They share the
    // expert readers, but have their own queue and completion count.
    std::deque<llama_moe_stream_ple_work> q_ple;
    size_t ple_pending = 0;
    std::vector<uint8_t> ple_buf;

    // Speculative loads (lookahead and hash prefetch) live in their OWN queue, drained only when
    // nothing is being waited on. With a single FIFO a wide prefetch delays the demand read a layer
    // is blocked on: measured at lookahead K=16, misses/remap fell 0.251 -> 0.172 but read latency
    // rose 1.69 -> 2.17 ms/slab and decode fell 8.45 -> 5.73 t/s. Separating them lets prefetch
    // width cost bandwidth (of which the drive has ~6x spare) instead of demand latency.
    std::deque<llama_moe_stream_work> q_spec;

    // cap the speculative backlog: a prefetch that is still queued when its layer arrives has done
    // nothing but reserve a slot and burn bandwidth
    size_t q_spec_max = 0;

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;

    bool debug = false;

    // GPU slot resolution (LLAMA_MOE_STREAM_GPU_SLOT): 0 off, 1 verify (the CPU remap still decides
    // and its answer is what reaches the GEMM, the kernel only checks itself against it), 2 publish
    // the table but insert no graph node - isolates the cost of the state buffer from the kernel.
    // LLAMA_MOE_STREAM_LRU: drop route hotness and evict purely by last use. Prerequisite for
    // moving residency to the GPU - hotness has no cheap kernel form, an argmin over 113 slots
    // does. Simulated at +11% misses.
    bool pure_lru     = false;

    int  gpu_slot     = 0;
    int64_t n_slot_chk = 0; // remap calls whose table the kernel checked
    int64_t n_slot_bad = 0; // of those, disagreements with the CPU - must stay 0
    int     n_slot_warn = 0; // warnings printed, capped

    struct llama_moe_stream_stats {
        int64_t n_calls     = 0; // remap invocations
        int64_t n_hit       = 0; // touched experts already resident or loading
        int64_t n_miss      = 0; // demand loads issued
        int64_t n_miss_cold = 0; // first-ever touch of an expert
        int64_t t_stall_us  = 0; // wait time in miss handling

        int64_t n_wave_calls     = 0; // wave-ids invocations (>= n_calls under multi-pass prefill)
        int64_t n_waves_run      = 0; // non-empty waves
        int64_t n_preload_issued = 0; // next-wave loads started during a wave's compute
        int64_t n_preload_ready  = 0; // wave experts already resident from the previous preload

        // Decode has no waves, so n_preload_ready above is unreachable there and reads as 0 - it
        // cannot say whether the lookahead prefetch is working. These two split a demand HIT by the
        // slot's state, which distinguishes the two failure modes:
        //   hit while LOADING  -> the prefetch picked the right expert but issued it too late
        //   miss (not counted here) -> the prefetch picked the wrong expert, or never issued
        int64_t n_hit_loading = 0; // hit on a slot still loading (a preload that did not land in time)
        int64_t n_hit_ready   = 0; // hit on a slot already resident
        int64_t t_stall_wave_us  = 0; // wait time in wave miss handling

        // Total wall time inside the streaming custom ops, stall included. These ops run on the CPU
        // as graph dependencies, so the GPU is idle for their duration and the time is on the
        // critical path. (op - stall) is therefore the CPU-side staging cost - planning waves,
        // picking victims, and emit_wave_slots' n_tok*n_ids id rewrite - as opposed to waiting on I/O.
        // Needed to answer what the ~92% of non-stall prefill actually is: Metal GEMM, or this.
        int64_t t_wave_op_us  = 0; // in llama_moe_stream_wave_ids  (prefill, multi-wave path)
        int64_t t_remap_op_us = 0; // in llama_moe_stream_remap     (decode, single-wave path)

        // Pair partitioning only. The graph fixes the per-wave pair chunk before the router runs, so
        // the slack it carries over the mean has to cover whatever imbalance the planner is left with
        // after LPT - and that slack is pure waste, the one cost the partition path still pays. This
        // is the measurement that sizes it: the worst (max wave load / mean - 1) seen, in percent.
        int64_t pair_over_max = 0;

        // worst wave load as a % of plan_pair_chunk. The margin to 100% IS the crash margin, and
        // unlike imbalance-over-mean it accounts for the n_tokens floor on the chunk.
        int64_t chunk_util_max = 0;

        // how often a hot expert had to be split across waves to fit the static chunk
        int64_t n_pair_splits = 0;

        // per-miss anatomy: SSD read vs Metal upload, and how many slabs were moved
        int64_t t_io_read_us   = 0;
        int64_t t_io_upload_us = 0;
        int64_t n_slabs_read   = 0;

        // read latency distribution; see MOE_STREAM_READ_BUCKET_US
        int64_t n_read_bucket[MOE_STREAM_READ_BUCKETS] = {0};
    };

    llama_moe_stream_stats stats;

    // Periodic delta dump (LLAMA_MOE_STREAM_STATS_MS=<ms>, unset = off).
    //
    // print_stats() alone cannot answer "how much of prefill is spent waiting for experts": it is
    // cumulative over the run, so prefill and decode are summed together, it is LLAMA_LOG_INFO which
    // llama-server filters out, and it only runs on clean shutdown - a pkill loses it entirely.
    // This dumps the deltas since the last dump, at WARN, on a wall-clock interval, so each line
    // covers one phase and carries the ratio that matters: stall time over elapsed time.
    int64_t                stats_dump_us    = 0; // interval, 0 = disabled
    int64_t                stats_t_last_us  = 0;
    llama_moe_stream_stats stats_prev;
    void maybe_dump_stats_locked();

    // internals
    void start_workers_locked();
    void worker_loop();
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);

    // multi-pass prefill helpers (called by llama_moe_stream_wave_ids, all under mtx)
    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n, uint32_t n_ids);
    void plan_pairs_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n); // wave 0: build the plan
    void emit_wave_pairs(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_pairs, int64_t chunk); // the 4 index rows
    void stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids); // make wave w resident + preload next
    void emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_tok); // write the slot ids
};

// Metal servicer entry point (LLAMA_MOE_STREAM_GPU_SLOT=3). Runs on Metal's listener queue with
// the GPU stalled, so it must do the reads and return. user_data is the llama_moe_stream.
void llama_moe_stream_service_gpu(void * user_data, void * state_host, int32_t layer);

// how many tokens a ubatch may carry for the GPU to own residency. Decode and speculative
// verification only: the kernel resolves the pair list serially in one thread, which is the right
// trade for a handful of pairs and the wrong one for a 4096-token prefill. Prefill keeps the CPU
// path, where one graph serves the whole ubatch and the split costs almost nothing anyway.
static const int32_t LLAMA_MOE_GPU_SLOT_MAX_TOKENS = 32;

// callback of the id-remapping custom op inserted by build_moe_ffn
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// remap for THIS layer (src a, as above) plus a lookahead prefetch for the next layer driven by
// this layer's router input (src b, f32 [n_embd] already multiplied by the next layer's gate_inp,
// i.e. b holds the next layer's predicted logits). userdata is a llama_moe_stream_lookahead.
void llama_moe_stream_remap_la(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata);

// Identity on the ubatch token ids, with a side effect: start the loads for every hash-routed
// layer. The hash layers take their tid2eid get_rows index from this op's output, which is what
// orders it before layer 0. Never waits.
void llama_moe_stream_prefetch_hash(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// callbacks of the multi-pass prefill custom ops inserted by build_moe_ffn when a ubatch touches
// more experts than the cache holds; each src[0] is the contiguous selected ids
//   wave_ids:   makes wave w's expert slice resident and emits slot ids (masked pairs park on a pool)
//   wave_mask:  emits 1.0 for pairs belonging to wave w, 0.0 otherwise
//   wave_pairs: same staging, but emits the index rows of wave w's dense pair list (partition path)
void llama_moe_stream_wave_ids  (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_mask (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_pairs(ggml_tensor * dst, int ith, int nth, void * userdata);

// index rows of the wave_pairs output, an I32 [chunk, LLAMA_MOE_PAIR_ROWS] tensor
enum llama_moe_pair_row {
    LLAMA_MOE_PAIR_TOK  = 0, // token index, to gather the GEMM input row
    LLAMA_MOE_PAIR_PAIR = 1, // flat pair index t*n_ids + k, to gather (weight_before_ffn) and scatter
    LLAMA_MOE_PAIR_SLOT = 2, // cache slot the GEMM indexes
    LLAMA_MOE_PAIR_EXP  = 3, // original expert id, for the biases and per-expert scales
    LLAMA_MOE_PAIR_ROWS = 4,
};
