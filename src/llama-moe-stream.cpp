#include "llama-moe-stream.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#include <sys/mman.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <mach/mach.h>
#endif
#include <chrono>
#include "llama-moe-lookahead.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <thread>
#include <system_error>
#include <atomic>
#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;
// Route-hotness halves every this many tokens. 1024, not the original 64: at 64 an expert accumulates
// only ~1.5 uses between halvings (256 experts, 6 picked per token), so counters sit at 0-3, cannot
// rank experts, and eviction degenerates into plain LRU. Measured on decode, 3 runs each:
//   64 -> 7.54 / 7.61 / 7.82 t/s, miss ~6.8%     1024 -> 9.06 / 8.85 / 9.11 t/s, miss ~5.4%
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 1024;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
#if defined(__APPLE__)
static const char * const MOE_STREAM_DIRECT_NAME = "F_NOCACHE";
#else
static const char * const MOE_STREAM_DIRECT_NAME = "O_DIRECT";
#endif

static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

// saturating increment - route-hotness counters accumulate over a whole run and must not wrap
static uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

// page-aligned allocation, required both for O_DIRECT reads and for Metal private-buffer uploads
static void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); staging must have
// room for len (+ 2*MOE_STREAM_DIRECT_ALIGN when direct). returns a pointer to the len bytes
// within staging, or nullptr on failure
static const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // no positional read primitive; serialize the seek+read pairs
    static std::mutex io_mtx;
    std::lock_guard<std::mutex> lock(io_mtx);
    try {
        file.seek(offs, SEEK_SET);
        file.read_raw(staging, len);
        return staging;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

// true iff all of the given exps tensors are this layer's cache tensors - guards against a second,
// non-streamed expert group on the same layer index (e.g. grovemoe chexps)
bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct) : n_slots(n_slots) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    use_direct_io = direct;

    if (const char * s = std::getenv("LLAMA_MOE_STREAM_STATS_MS")) {
        stats_dump_us = std::max<int64_t>(0, std::atoll(s))*1000;
    }

    // enough to keep every reader busy a few slabs deep, not so much that a prefetch is still
    // queued when its layer arrives
    q_spec_max = (size_t) this->n_io_threads * 8;
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_SPEC_MAX")) {
        q_spec_max = (size_t) std::max(0, atoi(s));
    }
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_WARM")) {
        warm_decode = atoi(s) != 0;
    }
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_INPLACE")) {
        inplace     = atoi(s) != 0;
        inplace_cpu = atoi(s) == 1;
    }
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
        q_ple.clear();
        q_spec.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    ggml_context * ctx = nullptr;
    for (auto & group : cache_bufs) {
        if (group.il == il && group.buft == buft) {
            ctx = group.ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*5,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        llama_moe_stream_buffer group;
        group.il   = il;
        group.buft = buft;
        group.ctx.reset(ctx);
        cache_bufs.emplace_back(std::move(group));
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_pending .resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->slot_spec    .resize(n_slots, 0);
        sl->route_hotness.resize(n_expert, 0);
        sl->decode_hot.resize(n_expert, 0);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);

    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::register_ple(const ggml_tensor * meta, uint16_t file_idx, size_t offs) {
    GGML_ASSERT(meta != nullptr);
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(ggml_n_dims(meta) == 2);
    GGML_ASSERT(meta->ne[0] > 0 && meta->ne[1] > 0);
    GGML_ASSERT(!ple.registered);

    ple.registered = true;
    ple.type       = meta->type;
    ple.file_idx   = file_idx;
    ple.offs       = offs;
    ple.row_ne     = meta->ne[0];
    ple.n_rows     = meta->ne[1];
    ple.row_size   = ggml_row_size(meta->type, meta->ne[0]);

    GGML_ASSERT(ple.row_size == meta->nb[1]);
    GGML_ASSERT(ple.row_size * (size_t) ple.n_rows == ggml_nbytes(meta));
}

void llama_moe_stream::read_ple_rows(const std::vector<int32_t> & rows, ggml_tensor * dst) {
    GGML_ASSERT(ple.registered && ple_file != nullptr);
    GGML_ASSERT(dst != nullptr && dst->type == ple.type);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[0] == ple.row_ne);
    GGML_ASSERT((size_t) ggml_nelements(dst) == (size_t) ple.row_ne * rows.size());

    const size_t nbytes = ple.row_size * rows.size();
    GGML_ASSERT(ggml_nbytes(dst) == nbytes);

    uint8_t * out_data = (uint8_t *) ggml_backend_tensor_get_host_ptr(dst);
    const bool direct = out_data != nullptr;

    // Read each unique row once. Sorting by file offset gives the buffered reader and SSD a less
    // hostile access pattern during large prefills; decode still consists of only a few rows.
    std::vector<std::pair<int32_t, size_t>> reads;
    std::vector<size_t> duplicate_of(rows.size(), SIZE_MAX);
    std::unordered_map<int32_t, size_t> first;
    first.reserve(rows.size());

    for (size_t i = 0; i < rows.size(); ++i) {
        const int32_t row = rows[i];
        if (row < 0 || (int64_t) row >= ple.n_rows) {
            throw std::runtime_error(format("PLE row %d is outside [0, %lld)", row, (long long) ple.n_rows));
        }
        const auto [it, inserted] = first.emplace(row, i);
        if (inserted) {
            reads.emplace_back(row, i);
        } else {
            duplicate_of[i] = it->second;
        }
    }
    std::sort(reads.begin(), reads.end(), [](const auto & a, const auto & b) { return a.first < b.first; });

    {
        std::unique_lock<std::mutex> lk(mtx);
        if (load_failed) {
            throw std::runtime_error("PLE row streaming: a previous model read failed");
        }
        GGML_ASSERT(ple_pending == 0 && q_ple.empty());

        if (!direct) {
            ple_buf.resize(nbytes);
            out_data = ple_buf.data();
        }
        ple_pending = reads.size();
        start_workers_locked();

        for (const auto & [row, i] : reads) {
            q_ple.push_back({ row, out_data + i*ple.row_size });
        }
        cv_work.notify_all();

        // Do not return while another reader still owns a pointer into dst or ple_buf. In
        // particular, an early I/O failure must not let graph teardown invalidate that pointer.
        cv_done.wait(lk, [&] { return ple_pending == 0; });
        if (load_failed) {
            throw std::runtime_error("PLE row streaming: model read failed");
        }
    }

    for (size_t i = 0; i < duplicate_of.size(); ++i) {
        if (duplicate_of[i] != SIZE_MAX) {
            memcpy(out_data + i*ple.row_size,
                   out_data + duplicate_of[i]*ple.row_size,
                   ple.row_size);
        }
    }

    if (!direct) {
        ggml_backend_tensor_set(dst, out_data, 0, nbytes);
    }
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    this->no_alloc = no_alloc;
    for (auto & group : cache_bufs) {
        ggml_backend_buffer_type_t buft = group.buft;
        ggml_context * ctx = group.ctx.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        group.buf.reset(buf);
        group.cap_slots = n_slots;
        group.released  = 0;
        group.split_slots = 0;

        LLAMA_LOG_DEBUG("%s: layer %d %12s expert cache size = %8.2f MiB (%u slots)\n",
                __func__, group.il, ggml_backend_buffer_name(buf),
                ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }

    LLAMA_LOG_INFO("%s: expert cache size = %8.2f MiB (%u slots per layer, %zu layer buffers)\n",
            __func__, size_bufs() / 1024.0 / 1024.0, n_slots, cache_bufs.size());
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

    open_all(use_direct_io);

    if (ple.registered) {
        if (ple.file_idx >= paths.size()) {
            throw std::runtime_error("PLE row streaming file index is out of range");
        }
        ple_file = std::make_unique<llama_file>(paths[ple.file_idx].c_str(), "rb", false);
        LLAMA_LOG_WARN("%s: PLE row streaming enabled: %.2f GiB table, %zu-byte rows, buffered parallel reads\n",
                __func__, (double) (ple.row_size * (size_t) ple.n_rows) / (1024.0*1024.0*1024.0), ple.row_size);
    }

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        bool ok = !files.empty() && files.front()->has_direct_io();
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true) != nullptr;
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: %s not usable, falling back to buffered streaming reads\n",
                    __func__, MOE_STREAM_DIRECT_NAME);
            use_direct_io = false;
            open_all(false);
        }
    }

    // Named rather than hardcoded: macOS reaches this through F_NOCACHE, not O_DIRECT, and a log
    // line claiming the wrong mechanism is worse than none - it cannot be told from a real bypass.
    if (use_direct_io) {
        LLAMA_LOG_WARN("%s: MoE expert streaming uses %s (page cache bypassed)\n",
                __func__, MOE_STREAM_DIRECT_NAME);
    }

    no_zerocopy = std::getenv("LLAMA_MOE_STREAM_NO_ZEROCOPY") != nullptr;

    // whether reads land in the cache slot directly or stage through a bounce buffer - worth a line
    // because it silently changes the per-miss cost and depends on the backend's buffer type
    for (const auto & sl : layers) {
        if (sl && !sl->weights.empty()) {
            const bool zc = !use_direct_io && !no_zerocopy &&
                    ggml_backend_tensor_get_host_ptr(sl->weights[0].cache) != nullptr;
            LLAMA_LOG_WARN("%s: MoE expert streaming reads %s\n",
                    __func__, zc ? "directly into the expert cache (no staging copy)"
                                 : "through a staging buffer");
            break;
        }
    }


    // one token drives ~one remap per streamed layer, so decaying every 64 tokens is
    //   64 * n_streamed_layers remap calls (computed once here, off the hot path)
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    // Eviction is hotness-with-decay: every hot_decay_interval remap calls, all counters halve, so
    // recent routing outweighs old routing. The 64-token constant has never been measured - and the
    // miss RATE is now the lever that matters, because the read path itself is close to the drive's
    // practical limit. LLAMA_MOE_STREAM_HOT_DECAY sweeps it without a rebuild.
    int64_t decay_tokens = MOE_STREAM_HOT_DECAY_TOKENS;
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_HOT_DECAY")) {
        decay_tokens = std::max<int64_t>(0, std::atoll(s));   // 0 = never decay (pure cumulative)
    }
    hot_decay_interval = decay_tokens * n_streamed;

}

// spawn the I/O thread pool on first use (from the remap callback, under mtx)
void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    reads_inflight.resize(n_io_threads);
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this, i]() { worker_loop((size_t) i); });
    }
}

// I/O worker: pops a reserved load, reads its expert slab(s) from the GGUF file into the cache
// slot, and marks the slot RESIDENT (or flags load_failed); stale/duplicate items are skipped
void llama_moe_stream::worker_loop(size_t worker_id) {
    // page-aligned staging (Metal private buffers require page-aligned source + page-multiple
    // length; O_DIRECT needs the extra head/tail slack for its aligned reads)
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(max_nb_expert + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&] {
            return shutting_down || !q_demand.empty() || !q_ple.empty() || !q_spec.empty();
        });
        if (shutting_down) {
            break;
        }

        // demand first, always. PLE rows are also blocking graph inputs, ahead of speculative work.
        if (q_demand.empty() && !q_ple.empty()) {
            const llama_moe_stream_ple_work w = q_ple.front();
            q_ple.pop_front();
            workers_active++;

            lk.unlock();
            const uint8_t * data = llama_moe_stream_pread(*ple_file, w.dst, ple.row_size,
                    ple.offs + (size_t) w.row*ple.row_size, /*direct =*/ false);
            lk.lock();
            GGML_ASSERT(workers_active > 0);
            workers_active--;

            if (data == nullptr) {
                load_failed = true;
            }
            GGML_ASSERT(ple_pending > 0);
            ple_pending--;
            cv_done.notify_all();
            continue;
        }

        // demand first, always: a layer is blocked on those, nothing is blocked on a prefetch
        llama_moe_stream_work w;
        if (!q_demand.empty()) {
            w = q_demand.front();
            q_demand.pop_front();
            w.demand = true;
        } else {
            w = q_spec.front();
            q_spec.pop_front();
            w.demand = false;
        }

        auto & sl = *w.sl;
        // no per-slot exclusion: several workers legitimately hold different slabs of the SAME slot
        // at once, which is the entire point. Staleness is still checked per slot.
        if (w.epoch != allocation_epoch ||
            w.slot < 0 || (uint32_t) w.slot >= sl.n_slots ||
            w.gen != sl.slot_gen[w.slot] ||
            sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
            sl.slot_expert[w.slot] != w.expert ||
            w.widx < 0 || (size_t) w.widx >= sl.weights.size()) {
            cv_done.notify_all(); // resize may be waiting for the last queued (stale) job
            continue; // stale item
        }

        workers_active++;
        reads_inflight[worker_id] = w;
        if (w.demand) {
            demand_inflight++;
        } else {
            stats.n_spec_dispatch++;
            stats.n_spec_with_demand += demand_inflight > 0 || ple_pending > 0;
        }

        lk.unlock();

        // Each work item reads one slab, so multiple workers can serve the same expert in parallel.
        // Read into the cache slot itself when the backend hands out a host pointer - on unified
        // memory the slot IS host memory, so staging then uploading is a pure extra copy of the
        // whole slab. The direct-io path keeps staging: it needs the head/tail slack for its
        // block-aligned reads, which would scribble outside the slot.
        const auto & wt = sl.weights[w.widx];
        uint8_t * dst = nullptr;
        if (!use_direct_io && !no_zerocopy) {
            auto * host = (uint8_t *) ggml_backend_tensor_get_host_ptr(wt.cache);
            dst = host ? host + (size_t) w.slot*wt.nb_expert : nullptr;
        }

        const int64_t t0 = ggml_time_us();
        const uint8_t * data = llama_moe_stream_pread(*files[wt.file_idx], dst ? dst : staging,
                wt.nb_expert, wt.offs + (size_t) w.expert*wt.nb_expert, use_direct_io);
        const int64_t t1 = ggml_time_us();
        const bool ok = data != nullptr;
        if (ok && dst == nullptr) {
            ggml_backend_tensor_set(wt.cache, data, (size_t) w.slot*wt.nb_expert, wt.nb_expert);
        }
        const int64_t t2 = ggml_time_us();

        lk.lock();

        GGML_ASSERT(workers_active > 0);
        workers_active--;
        // Read the live flag: a formerly speculative read can become demand while IO runs.
        if (reads_inflight[worker_id].demand) {
            GGML_ASSERT(demand_inflight > 0);
            demand_inflight--;
        }
        reads_inflight[worker_id] = {};

        stats.t_io_read_us   += t1 - t0;
        stats.t_io_upload_us += ok ? t2 - t1 : 0;
        stats.n_slabs_read   += 1;
        stats.n_bytes_read   += ok ? wt.nb_expert : 0;

        {
            int b = 0;
            while (b < MOE_STREAM_READ_BUCKETS - 1 && (t1 - t0) >= MOE_STREAM_READ_BUCKET_US[b]) {
                b++;
            }
            stats.n_read_bucket[b]++;
        }

        if (w.epoch != allocation_epoch || (uint32_t) w.slot >= sl.n_slots) {
            // A resize cannot normally race an active reader because resize_slots() waits for the
            // active count to reach zero. Keep this guard so a future cancellation path cannot
            // publish through stale metadata.
        } else if (!ok) {
            load_failed = true;
            sl.slot_pending[w.slot] = 0;
        } else if (w.gen == sl.slot_gen[w.slot] && sl.slot_pending[w.slot] > 0) {
            // the LAST slab to land publishes the slot; until then it stays LOADING, so no consumer
            // can observe a half-filled expert
            if (--sl.slot_pending[w.slot] == 0) {
                sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            }
        }
        cv_done.notify_all();
    }
    lk.unlock();

    moe_aligned_free(staging);
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when no candidate exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the slot's prior occupant, bump slot_gen (so any
// in-flight load for the old occupant is recognized as stale), and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot.erase(sl.slot_expert[slot]);
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_spec[slot]   = 0;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
}

// A grow leaves its new slots empty, and they would otherwise fill only as decode misses on them -
// each one a demand read on the critical path. The previous decode phase is the best guess at what
// the next one wants, so queue its hottest non-resident experts into those slots. EMPTY slots only,
// never an eviction, and at speculative priority: demand still goes first, and a demand for an
// expert already warming promotes its queued read instead of issuing a second one.
void llama_moe_stream::warm_decode_locked() {
    start_workers_locked();

    size_t n_queued = 0;
    size_t n_layers = 0;
    std::vector<int32_t> cand;
    for (auto & slp : layers) {
        if (!slp) {
            continue;
        }
        auto & sl = *slp;

        uint32_t n_empty = 0;
        for (uint32_t s = 0; s < sl.n_slots; ++s) {
            n_empty += sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY;
        }
        if (n_empty == 0) {
            continue;
        }

        cand.clear();
        for (uint32_t e = 0; e < sl.n_expert; ++e) {
            if (sl.decode_hot[e] > 0 && sl.expert_slot.find((int32_t) e) == sl.expert_slot.end()) {
                cand.push_back((int32_t) e);
            }
        }
        std::stable_sort(cand.begin(), cand.end(), [&](int32_t x, int32_t y) {
            return sl.decode_hot[x] > sl.decode_hot[y];
        });
        if (cand.size() > n_empty) {
            cand.resize(n_empty);
        }

        uint32_t s = 0;
        for (const int32_t e : cand) {
            while (s < sl.n_slots && sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_EMPTY) {
                s++;
            }
            if (s >= sl.n_slots) {
                break;
            }
            reserve_slot_locked(sl, e, (int32_t) s);
            sl.slot_pending[s] = (uint8_t) sl.weights.size();
            for (size_t wi = 0; wi < sl.weights.size(); wi++) {
                q_spec.push_back({ &sl, e, (int32_t) s, (int32_t) wi, sl.slot_gen[s], allocation_epoch });
            }
            stats.n_warm_issued++;
            n_queued++;
        }
        n_layers += cand.empty() ? 0 : 1;
    }

    if (n_queued > 0) {
        cv_work.notify_all();
        LLAMA_LOG_WARN("%s: queued %zu decode-hot experts into empty slots across %zu layers\n",
                __func__, n_queued, n_layers);
    }
}

// A speculative hit already has exactly one work item per slab. Move the queued items to the
// demand queue and mark in-flight items as required. Re-enqueuing every slab and resetting
// slot_pending can publish the slot after duplicate completions while another slab is still unread.
void llama_moe_stream::promote_slot_locked(llama_moe_stream_layer & sl, int32_t slot) {
    for (auto & w : reads_inflight) {
        if (w.sl == &sl && w.slot == slot && w.gen == sl.slot_gen[slot] &&
                w.epoch == allocation_epoch && !w.demand) {
            w.demand = true;
            demand_inflight++;
        }
    }
    bool moved = false;
    for (auto it = q_spec.begin(); it != q_spec.end();) {
        if (it->sl == &sl && it->slot == slot && it->gen == sl.slot_gen[slot]) {
            q_demand.push_back(*it);
            it = q_spec.erase(it);
            moved = true;
        } else {
            ++it;
        }
    }
    if (moved) {
        cv_work.notify_all();
    }
}

size_t llama_moe_stream::size_bufs() const {
    size_t size = 0;
    for (const auto & group : cache_bufs) {
        if (group.buf) {
            size += ggml_backend_buffer_get_size(group.buf.get()) - group.released;
        }
    }
    return size;
}

size_t llama_moe_stream::released_bytes() const {
    size_t size = 0;
    for (const auto & group : cache_bufs) {
        size += group.released;
    }
    return size;
}

uint64_t llama_moe_stream::free_bytes() {
#if defined(__APPLE__)
    static const mach_port_t host = mach_host_self();
    vm_statistics64_data_t v;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t) &v, &count) != KERN_SUCCESS ||
            host_page_size(host, &page) != KERN_SUCCESS) {
        return 0;
    }
    return (uint64_t) v.free_count*page;
#else
    return 0;
#endif
}

// Wired memory, sampled every 2 ms, until it held within 16 MiB for settle_ms (or timeout_ms).
static uint64_t moe_stream_wired_settled(int settle_ms, int timeout_ms) {
    const int64_t t0 = ggml_time_us();
    uint64_t last = llama_moe_stream::wired_bytes();
    int64_t  t_change = t0;
    while (ggml_time_us() - t0 < (int64_t) timeout_ms*1000) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        const uint64_t w = llama_moe_stream::wired_bytes();
        if (w + (16u << 20) < last || last + (16u << 20) < w) {
            last = w;
            t_change = ggml_time_us();
        } else if (ggml_time_us() - t_change >= (int64_t) settle_ms*1000) {
            break;
        }
    }
    return last;
}

double llama_moe_stream::wait_unwired(uint64_t wired0, uint64_t bytes, int timeout_ms) {
    if (wired0 == 0 || bytes == 0) {
        return 0.0;
    }
    // Until the driver drops its wiring a page cannot be freed: released then, it stays with the
    // allocation, unmapped, and later gets compressed or swapped (measured: anonymous memory rose by
    // the whole tail). Wait for nearly all of it to leave the wired pool, then for the count to settle.
    const uint64_t drop   = bytes/20*19;
    const uint64_t target = wired0 > drop ? wired0 - drop : 0;
    const int64_t  t0     = ggml_time_us();
    while (wired_bytes() > target) {
        if (ggml_time_us() - t0 >= (int64_t) timeout_ms*1000) {
            LLAMA_LOG_WARN("%s: %.0f MiB still wired after %d ms; releasing anyway\n", __func__, bytes/1048576.0, timeout_ms);
            return (ggml_time_us() - t0)/1000.0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    moe_stream_wired_settled(250, 1000);
    return (ggml_time_us() - t0)/1000.0;
}

uint64_t llama_moe_stream::wired_bytes() {
#if defined(__APPLE__)
    static const mach_port_t host = mach_host_self();
    vm_statistics64_data_t v;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t) &v, &count) != KERN_SUCCESS ||
            host_page_size(host, &page) != KERN_SUCCESS) {
        return 0;
    }
    return (uint64_t) v.wire_count*page;
#else
    return 0;
#endif
}

uint64_t llama_moe_stream::task_resident_bytes() {
#if defined(__APPLE__)
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
#else
    return 0;
#endif
}

bool llama_moe_stream::can_resize_inplace(uint32_t slots) const {
    if (!inplace) {
        return false;
    }
    bool any = false;
    for (const auto & group : cache_bufs) {
        if (!group.buf) {
            continue;
        }
        if (group.cap_slots < slots) {
            return false;
        }
        bool metal = false;
#ifdef GGML_USE_METAL
        metal = ggml_backend_metal_buffer_has_views(group.buf.get());
#endif
        if (!metal && !inplace_cpu) {
            return false;
        }
        if (metal && slots < n_slots && slots != group.split_slots) {
            return false;
        }
        any = true;
    }
    return any;
}

uint64_t llama_moe_stream::bytes_per_slot() const {
    uint64_t size = 0;
    for (const auto & sl : layers) {
        if (!sl) {
            continue;
        }
        for (const auto & wt : sl->weights) {
            size += wt.nb_expert;
        }
    }
    return size;
}

uint32_t llama_moe_stream::slots_for_budget(uint64_t budget, uint32_t n_slots_min) const {
    const uint64_t per_slot = bytes_per_slot();
    if (budget == 0 || per_slot == 0) {
        return n_slots;
    }

    uint32_t n_expert = UINT32_MAX;
    for (const auto & sl : layers) {
        if (sl) {
            n_expert = std::min(n_expert, sl->n_expert);
        }
    }
    if (n_expert == UINT32_MAX || n_expert <= 1) {
        return n_slots;
    }

    uint32_t lo = std::min(std::max(1u, n_slots_min), n_expert - 1);
    uint32_t hi = n_expert - 1;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1)/2;
        if (allocation_size(mid) <= budget) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

uint64_t llama_moe_stream::allocation_size(uint32_t slots, uint64_t * max_layer) const {
    std::vector<uint64_t> by_layer(layers.size(), 0);
    uint64_t total = 0;
    for (const auto & group : cache_bufs) {
        const size_t align = ggml_backend_buft_get_alignment(group.buft);
        uint64_t bytes = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(group.ctx.get()); t;
                t = ggml_get_next_tensor(group.ctx.get(), t)) {
            ggml_tensor resized = *t;
            resized.ne[2] = slots;
            resized.nb[3] = resized.nb[2] * slots;
            bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(group.buft, &resized), align);
        }
        total += bytes;
        by_layer.at(group.il) += bytes;
    }
    if (max_layer) {
        *max_layer = by_layer.empty() ? 0 : *std::max_element(by_layer.begin(), by_layer.end());
    }
    return total;
}

uint32_t llama_moe_stream::slots_for_reclaimed(uint64_t reclaimed, uint64_t margin) const {
    if (reclaimed <= margin || cache_bufs.empty()) {
        return n_slots;
    }
    uint32_t hi = UINT32_MAX;
    for (const auto & sl : layers) {
        if (sl) {
            hi = std::min(hi, sl->n_expert - 1);
        }
    }
    const uint64_t base = allocation_size(n_slots);
    uint32_t lo = n_slots;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo + 1)/2;
        uint64_t scratch = 0;
        const uint64_t extra = allocation_size(mid, &scratch) - base;
        const uint64_t usable = reclaimed - margin;
        if (extra <= usable && scratch <= usable - extra) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

namespace {

struct moe_stream_replacement {
    uint32_t split_slots = 0; // tails [split_slots, new slots) allocated as their own memory objects
    llama_moe_stream_buffer * group = nullptr;
    ggml_context_ptr          ctx;
    ggml_backend_buffer_ptr   buf;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> tensors;
};

// The manager lock is held and no reader is active. Allocate and populate every backing group of
// one layer before publishing any pointer, so an allocation failure leaves that layer untouched.
struct moe_stream_copy_job {
    uint8_t *       dst;
    const uint8_t * src;
    size_t          n;
};

// Retained experts are copied host to host on a transition that cannot resize in place. The copy is
// bound by first touch of the new buffer, not memcpy bandwidth, so it runs on the calling thread.
static void moe_stream_copy(const std::vector<moe_stream_copy_job> & jobs) {
    for (const auto & j : jobs) {
        memcpy(j.dst, j.src, j.n);
    }
}

// The experts a layer keeps at new_slots: the hottest (or most recently used) when shrinking,
// all of them in slot order when growing.
// Page-aligned [begin, end) of a cache tensor's slots [from, to), rounded inward: the pages that hold
// nothing but those slots. The split allocation and the later release must agree on it exactly.
static std::pair<size_t, size_t> moe_stream_tail_range(size_t off, size_t nb_expert, uint32_t from, uint32_t to, size_t page) {
    const size_t b = (off + (size_t) from*nb_expert + page - 1)/page*page;
    const size_t e = (off + (size_t) to*nb_expert)/page*page;
    return { b, e > b ? e : b };
}

// Allocates a group's replacement tensors in a Metal buffer whose per-tensor tails - slots [from, to) -
// are their own memory objects. Offsets follow ggml_backend_alloc_ctx_tensors: in order, each at the
// buffer type's alignment. Returns null (the caller allocates normally) where this does not apply.
static ggml_backend_buffer_t moe_stream_alloc_split(ggml_context * ctx, ggml_backend_buffer_type_t buft, uint32_t from, uint32_t to) {
#ifdef GGML_USE_METAL
    const size_t align = ggml_backend_buft_get_alignment(buft);
    const size_t page  = (size_t) sysconf(_SC_PAGESIZE);
    std::vector<std::pair<ggml_tensor *, size_t>> placed;
    std::vector<size_t> offs;
    std::vector<size_t> sizes;
    size_t off = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        off = GGML_PAD(off, align);
        placed.emplace_back(t, off);
        const auto [b, e] = moe_stream_tail_range(off, t->nb[2], from, to, page);
        if (e > b) {
            offs.push_back(b);
            sizes.push_back(e - b);
        }
        off += ggml_backend_buft_get_alloc_size(buft, t);
    }
    ggml_backend_buffer_t buf = ggml_backend_metal_buffer_type_alloc_split(buft, off, offs.data(), sizes.data(), (int) offs.size());
    if (buf == nullptr) {
        return nullptr;
    }
    auto * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    for (const auto & [t, o] : placed) {
        if (ggml_backend_tensor_alloc(buf, t, base + o) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            return nullptr;
        }
    }
    return buf;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(buft);
    GGML_UNUSED(from);
    GGML_UNUSED(to);
    return nullptr;
#endif
}

static std::vector<uint32_t> moe_stream_retained_slots(llama_moe_stream_layer & sl, uint32_t new_slots) {
    std::vector<uint32_t> resident;
    resident.reserve(sl.n_slots);
    for (uint32_t s = 0; s < sl.n_slots; ++s) {
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT && sl.slot_expert[s] >= 0) {
            resident.push_back(s);
        }
    }

    if (resident.size() > new_slots) {
        std::stable_sort(resident.begin(), resident.end(), [&](uint32_t a, uint32_t b) {
            const int32_t ea = sl.slot_expert[a];
            const int32_t eb = sl.slot_expert[b];
            const uint32_t ha = ea >= 0 ? sl.route_hotness[ea] : 0;
            const uint32_t hb = eb >= 0 ? sl.route_hotness[eb] : 0;
            return ha != hb ? ha > hb : sl.slot_last_use[a] > sl.slot_last_use[b];
        });
        resident.resize(new_slots);
    } else {
        // Growing keeps the old slot order. Besides minimizing metadata churn, this makes a
        // grow/shrink transition deterministic when every resident expert fits.
        std::sort(resident.begin(), resident.end());
    }

    return resident;
}

// Every capacity-dependent prefill plan is rebuilt by the next graph/callback invocation.
static void moe_stream_reset_plans(llama_moe_stream_layer & sl) {
    sl.plan_capacity  = 0;
    sl.plan_n_waves   = 0;
    sl.plan_next_wave = -1;
    sl.plan_pair_chunk = 0;
    sl.plan_waves_want = 0;
    sl.expert_wave.clear();
    sl.plan_pool.clear();
    sl.pool_used.clear();
    sl.wave_first.clear();
    sl.wave_count.clear();
    sl.plan_pair.clear();
    sl.pair_count.clear();
}

static bool moe_stream_can_release(ggml_backend_buffer_t buf) {
#ifdef GGML_USE_METAL
    return ggml_backend_metal_buffer_has_views(buf);
#else
    GGML_UNUSED(buf);
    return false;
#endif
}

// compared by identity: the CPU buffer type reports no device
static bool moe_stream_is_cpu_buffer(ggml_backend_buffer_t buf) {
    return ggml_backend_buffer_get_type(buf) == ggml_backend_cpu_buffer_type();
}

// In-place resizing needs every group of the layer to hold new_slots already and host-visible
// memory for the tail moves. Metal shared buffers get their views re-wrapped and their tails
// released; a CPU-device buffer (tests, CPU-only runs) is resized logically and stays resident.
static bool moe_stream_inplace_fits(const llama_moe_stream & mgr, const llama_moe_stream_layer & sl, uint32_t new_slots) {
    bool any = false;
    for (const auto & group : mgr.cache_bufs) {
        if (group.il != sl.il) {
            continue;
        }
        ggml_backend_buffer_t buf = group.buf.get();
        if (buf == nullptr || group.cap_slots < new_slots || ggml_backend_buffer_get_base(buf) == nullptr) {
            return false;
        }
        if (!moe_stream_can_release(buf) && !(mgr.inplace_cpu && moe_stream_is_cpu_buffer(buf))) {
            return false;
        }
        // on Metal a shrink must land on the split point: elsewhere it would free part of an object
        if (moe_stream_can_release(buf) && new_slots < sl.n_slots && new_slots != group.split_slots) {
            return false;
        }
        any = true;
    }
    return any;
}

#ifdef GGML_USE_METAL
static std::mutex moe_stream_views_mtx; // guards pending_release while layers re-wrap in parallel

// Re-wrap a group's GPU views around its live slots and, on a shrink, return the tails past them
// to the system. Tensors keep the offsets of the full-capacity layout, so each tail sits mid-buffer:
// the views are the complement of the tails, page-aligned because each tail is rounded inward.
// A view spanning a released range would fault it back in, so the views change before the release.
static bool moe_stream_apply_views(llama_moe_stream & mgr, llama_moe_stream_buffer & group, uint32_t live_slots, bool release) {
    ggml_backend_buffer_t buf = group.buf.get();
    auto * base = (uint8_t *) ggml_backend_buffer_get_base(buf);
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t size = (ggml_backend_buffer_get_size(buf) + page - 1)/page*page;

    std::vector<std::pair<size_t, size_t>> tails;
    for (ggml_tensor * t = ggml_get_first_tensor(group.ctx.get()); t; t = ggml_get_next_tensor(group.ctx.get(), t)) {
        const size_t off = (size_t) ((uint8_t *) t->data - base);
        const auto [b, e] = moe_stream_tail_range(off, t->nb[2], live_slots, group.cap_slots, page);
        if (e > b) {
            tails.emplace_back(b, e);
        }
    }
    std::sort(tails.begin(), tails.end());

    std::vector<size_t> offs;
    std::vector<size_t> sizes;
    size_t at = 0;
    for (const auto & [b, e] : tails) {
        if (b > at) {
            offs.push_back(at);
            sizes.push_back(b - at);
        }
        at = std::max(at, e);
    }
    if (size > at) {
        offs.push_back(at);
        sizes.push_back(size - at);
    }

    const auto install = mgr.resize_set_views ? mgr.resize_set_views : ggml_backend_metal_buffer_set_views;
    if (!install(buf, offs.data(), sizes.data(), (int) offs.size())) {
        // A failed shrink can retain larger old views. A failed grow after a shrink cannot:
        // those views exclude the new tail. The caller must roll back the logical layout.
        LLAMA_LOG_WARN("%s: layer %d: could not re-wrap %zu views; %s\n",
                __func__, group.il, offs.size(), release ? "keeping the tail resident" : "rolling back growth");
        return release;
    }

    // On a shrink the tails are returned only after the driver unwired them (resize_slots waits for
    // that across all layers); on a grow the remaining tails were released earlier.
    size_t released = 0;
    std::lock_guard<std::mutex> lock(moe_stream_views_mtx);
    for (const auto & [b, e] : tails) {
        if (release) {
            mgr.pending_release.emplace_back(base + b, e - b);
        }
        released += e - b;
    }
    group.released = released;
    return true;
}
#endif

// Resize within the existing allocation: no replacement buffer and no copy, except for retained
// experts sitting in a shrinking tail, which move into free head slots. A grow leaves the new
// slots empty for demand loads and the warm-up. Same retention ranking as the copying path.
static bool moe_stream_resize_layer_inplace(llama_moe_stream & mgr, llama_moe_stream_layer & sl, uint32_t new_slots) {
    const uint32_t old_slots = sl.n_slots;

    std::vector<uint32_t> resident = moe_stream_retained_slots(sl, new_slots);
    std::sort(resident.begin(), resident.end());

    std::vector<int32_t>  slot_expert(new_slots, -1);
    std::vector<uint8_t>  slot_state(new_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
    std::vector<uint8_t>  slot_pending(new_slots, 0);
    std::vector<uint64_t> slot_gen(new_slots, 1);
    std::vector<int64_t>  slot_last_use(new_slots, 0);
    std::vector<uint8_t>  keep(new_slots, 0);
    std::unordered_map<int32_t, int32_t> expert_slot;
    expert_slot.reserve(resident.size());

    // every surviving slot index advances its generation, so no stale job can publish into it
    for (uint32_t d = 0; d < new_slots; ++d) {
        slot_gen[d] = (d < old_slots ? sl.slot_gen[d] : 0) + 1;
    }

    std::vector<uint8_t> taken(new_slots, 0);
    for (const uint32_t s : resident) {
        if (s < new_slots) {
            taken[s] = 1;
        }
    }
    std::vector<std::pair<uint32_t, uint32_t>> moves;
    uint32_t free_at = 0;
    for (const uint32_t src : resident) {
        uint32_t dst = src;
        if (src >= new_slots) {
            while (free_at < new_slots && taken[free_at]) {
                free_at++;
            }
            GGML_ASSERT(free_at < new_slots); // at most new_slots experts are retained
            dst = free_at;
            taken[dst] = 1;
            moves.emplace_back(src, dst);
            slot_gen[dst] = std::max(slot_gen[dst], sl.slot_gen[src] + 1);
        }
        const int32_t expert = sl.slot_expert[src];
        slot_expert[dst]    = expert;
        slot_state[dst]     = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        slot_last_use[dst]  = sl.slot_last_use[src];
        expert_slot[expert] = (int32_t) dst;
    }

    std::vector<moe_stream_copy_job> jobs;
    for (auto & group : mgr.cache_bufs) {
        if (group.il != sl.il) {
            continue;
        }
        for (ggml_tensor * t = ggml_get_first_tensor(group.ctx.get()); t; t = ggml_get_next_tensor(group.ctx.get(), t)) {
            if (t->ne[2] != (int64_t) old_slots || t->ne[3] != 1) {
                LLAMA_LOG_ERROR("%s: unexpected cache tensor shape for %s\n", __func__, t->name);
                return false;
            }
            auto * host = (uint8_t *) ggml_backend_tensor_get_host_ptr(t);
            if (host == nullptr && ggml_backend_buffer_is_host(t->buffer)) {
                host = (uint8_t *) t->data; // CPU buffers expose no host-pointer hook
            }
            if (host == nullptr) {
                LLAMA_LOG_ERROR("%s: in-place resize needs host-visible cache tensors (%s)\n", __func__, t->name);
                return false;
            }
            const size_t nb_expert = t->nb[2];
            for (const auto & [src, dst] : moves) {
                jobs.push_back({ host + (size_t) dst*nb_expert, host + (size_t) src*nb_expert, nb_expert });
            }
        }
    }

    // nothing below can fail: the moves read tail slots, so they precede the release
    const int64_t t_copy = ggml_time_us();
    moe_stream_copy(jobs);
    mgr.stats.t_resize_copy_us += ggml_time_us() - t_copy;

    for (auto & group : mgr.cache_bufs) {
        if (group.il != sl.il) {
            continue;
        }
        for (ggml_tensor * t = ggml_get_first_tensor(group.ctx.get()); t; t = ggml_get_next_tensor(group.ctx.get(), t)) {
            t->ne[2] = new_slots;
            t->nb[3] = t->nb[2]*(size_t) new_slots;
        }
#ifdef GGML_USE_METAL
        if (moe_stream_can_release(group.buf.get())) {
            mgr.pending_views.emplace_back(&group, new_slots, new_slots < old_slots); // re-wrapped by resize_slots
        }
#endif
    }

    sl.n_slots       = new_slots;
    sl.slot_expert   = std::move(slot_expert);
    sl.slot_state    = std::move(slot_state);
    sl.slot_pending  = std::move(slot_pending);
    sl.slot_gen      = std::move(slot_gen);
    sl.slot_last_use = std::move(slot_last_use);
    sl.slot_spec     .assign(new_slots, 0); // observability only: a resize drops pending marks
    sl.expert_slot   = std::move(expert_slot);
    sl.keep = std::move(keep);
    sl.demand_slots.clear();
    moe_stream_reset_plans(sl);

    mgr.n_inplace_layers++;
    return true;
}

static bool moe_stream_resize_layer(
        llama_moe_stream & mgr,
        llama_moe_stream_layer & sl,
        uint32_t new_slots) {
    if (mgr.inplace && moe_stream_inplace_fits(mgr, sl, new_slots)) {
        return moe_stream_resize_layer_inplace(mgr, sl, new_slots);
    }

    std::vector<uint32_t> resident = moe_stream_retained_slots(sl, new_slots);

    std::vector<int32_t>  slot_expert(new_slots, -1);
    std::vector<uint8_t>  slot_state(new_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
    std::vector<uint8_t>  slot_pending(new_slots, 0);
    std::vector<uint64_t> slot_gen(new_slots, 1);
    std::vector<int64_t>  slot_last_use(new_slots, 0);
    std::vector<uint8_t>  keep(new_slots, 0);
    std::unordered_map<int32_t, int32_t> expert_slot;
    expert_slot.reserve(resident.size());

    std::vector<std::pair<uint32_t, uint32_t>> copies;
    copies.reserve(resident.size());
    for (uint32_t dst = 0; dst < resident.size(); ++dst) {
        const uint32_t src = resident[dst];
        const int32_t expert = sl.slot_expert[src];
        slot_expert[dst]   = expert;
        slot_state[dst]    = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        slot_gen[dst]      = sl.slot_gen[src] + 1;
        slot_last_use[dst] = sl.slot_last_use[src];
        expert_slot[expert] = (int32_t) dst;
        copies.emplace_back(src, dst);
    }

    std::vector<moe_stream_replacement> replacements;
    std::vector<moe_stream_copy_job>    jobs;
    for (auto & group : mgr.cache_bufs) {
        if (group.il != sl.il) {
            continue;
        }

        size_t n_tensors = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(group.ctx.get()); t;
                t = ggml_get_next_tensor(group.ctx.get(), t)) {
            n_tensors++;
        }

        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(n_tensors + 1),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };

        moe_stream_replacement replacement;
        replacement.group = &group;
        replacement.ctx.reset(ggml_init(params));
        if (!replacement.ctx) {
            LLAMA_LOG_ERROR("%s: failed to create replacement metadata for layer %d\n", __func__, sl.il);
            return false;
        }

        for (ggml_tensor * old = ggml_get_first_tensor(group.ctx.get()); old;
                old = ggml_get_next_tensor(group.ctx.get(), old)) {
            if (old->ne[2] != (int64_t) sl.n_slots || old->ne[3] != 1) {
                LLAMA_LOG_ERROR("%s: unexpected cache tensor shape for %s\n", __func__, old->name);
                return false;
            }
            ggml_tensor * fresh = ggml_new_tensor_3d(
                    replacement.ctx.get(), old->type, old->ne[0], old->ne[1], new_slots);
            ggml_set_name(fresh, old->name);
            replacement.tensors.emplace_back(old, fresh);
        }

        const int64_t t_alloc = ggml_time_us();
        // In in-place mode a grow splits each tensor's new tail off into its own memory objects, so
        // a later shrink back to today's size can return it whole.
        if (mgr.inplace && new_slots > sl.n_slots) {
            replacement.buf.reset(moe_stream_alloc_split(replacement.ctx.get(), group.buft, sl.n_slots, new_slots));
            replacement.split_slots = replacement.buf ? sl.n_slots : 0;
        }
        if (!replacement.buf) {
            replacement.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(
                    replacement.ctx.get(), group.buft));
        }
        mgr.stats.t_resize_alloc_us += ggml_time_us() - t_alloc;
        if (!replacement.buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate layer %d replacement cache (%u slots)\n",
                    __func__, sl.il, new_slots);
            return false;
        }
        ggml_backend_buffer_set_usage(replacement.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (const auto & [old, fresh] : replacement.tensors) {
            if (old->extra != nullptr || fresh->extra != nullptr) {
                LLAMA_LOG_ERROR("%s: tensor-specific backend metadata is not safely replaceable (%s)\n",
                        __func__, old->name);
                return false;
            }

            const size_t nb_expert = old->nb[2];
            GGML_ASSERT(nb_expert * (size_t) old->ne[2] == ggml_nbytes(old));
            GGML_ASSERT(nb_expert * (size_t) fresh->ne[2] == ggml_nbytes(fresh));

            const auto * src_host = (const uint8_t *) ggml_backend_tensor_get_host_ptr(old);
            auto *       dst_host = (uint8_t *) ggml_backend_tensor_get_host_ptr(fresh);
            if (src_host && dst_host) {
                for (const auto & [src, dst] : copies) {
                    jobs.push_back({ dst_host + (size_t) dst*nb_expert, src_host + (size_t) src*nb_expert, nb_expert });
                }
            } else {
                std::vector<uint8_t> staging(nb_expert);
                for (const auto & [src, dst] : copies) {
                    ggml_backend_tensor_get(old, staging.data(), (size_t) src*nb_expert, nb_expert);
                    ggml_backend_tensor_set(fresh, staging.data(), (size_t) dst*nb_expert, nb_expert);
                }
            }
        }

        replacements.emplace_back(std::move(replacement));
    }

    // every replacement buffer exists before any byte moves; publication below needs them all filled
    const int64_t t_copy = ggml_time_us();
    moe_stream_copy(jobs);
    mgr.stats.t_resize_copy_us += ggml_time_us() - t_copy;

    if (replacements.empty()) {
        LLAMA_LOG_ERROR("%s: layer %d has no backing allocation\n", __func__, sl.il);
        return false;
    }

    // Publication is non-throwing. Stable tensor objects are retained because model layers and
    // donor MTP graphs keep pointers to them; only their shapes and backing allocations change.
    std::vector<ggml_backend_buffer_ptr> old_bufs;
    old_bufs.reserve(replacements.size());
    for (auto & replacement : replacements) {
        for (auto & [old, fresh] : replacement.tensors) {
            old->buffer = fresh->buffer;
            old->data   = fresh->data;
            old->extra  = fresh->extra;
            memcpy(old->ne, fresh->ne, sizeof(old->ne));
            memcpy(old->nb, fresh->nb, sizeof(old->nb));

            fresh->buffer = nullptr;
            fresh->data   = nullptr;
            fresh->extra  = nullptr;
        }
        old_bufs.emplace_back(std::move(replacement.group->buf));
        replacement.group->buf = std::move(replacement.buf);
        replacement.group->cap_slots = new_slots;
        replacement.group->released  = 0;
        replacement.group->split_slots = replacement.split_slots;
    }
    // nothing references the old buffers after publication; release them here so it can be timed
    const int64_t t_free = ggml_time_us();
    old_bufs.clear();
    mgr.stats.t_resize_free_us += ggml_time_us() - t_free;

    sl.n_slots       = new_slots;
    sl.slot_expert   = std::move(slot_expert);
    sl.slot_state    = std::move(slot_state);
    sl.slot_pending  = std::move(slot_pending);
    sl.slot_gen      = std::move(slot_gen);
    sl.slot_last_use = std::move(slot_last_use);
    sl.slot_spec     .assign(new_slots, 0); // observability only: a resize drops pending marks
    sl.expert_slot   = std::move(expert_slot);
    sl.keep = std::move(keep);
    sl.demand_slots.clear();

    moe_stream_reset_plans(sl);

    return true;
}

} // namespace

bool llama_moe_stream::resize_slots(uint32_t new_slots) {
    if (new_slots == n_slots) {
        return true;
    }
    if (no_alloc) {
        LLAMA_LOG_ERROR("%s: dynamic cache resizing requires allocated CPU-managed slots\n", __func__);
        return false;
    }
    for (const auto & sl : layers) {
        if (sl && (new_slots == 0 || new_slots >= sl->n_expert)) {
            LLAMA_LOG_ERROR("%s: invalid slot count %u for %u experts\n",
                    __func__, new_slots, sl->n_expert);
            return false;
        }
    }

    std::unique_lock<std::mutex> lk(mtx);
    if (resizing) {
        LLAMA_LOG_ERROR("%s: cache resize already in progress\n", __func__);
        return false;
    }
    if (load_failed || shutting_down) {
        return false;
    }
    // Reserve bookkeeping before any publication; no allocation may throw after a layer commits.
    std::vector<llama_moe_stream_layer *> migrated;
    migrated.reserve(layers.size());
    resizing = true;
    struct resize_guard {
        llama_moe_stream & mgr;
        ~resize_guard() {
            mgr.resizing = false;
            mgr.cv_work.notify_all();
            mgr.cv_done.notify_all();
        }
    } guard{*this};

    // Demand/PLE work belongs to a submitted graph and must complete. Speculative prefetch is safe
    // to cancel; after its in-flight slabs settle, any incomplete reservation is discarded.
    q_spec.clear();
    cv_done.wait(lk, [&] {
        return q_demand.empty() && q_ple.empty() && ple_pending == 0 && workers_active == 0;
    });
    if (load_failed) {
        return false;
    }

    for (auto & sl : layers) {
        if (!sl) {
            continue;
        }
        for (uint32_t s = 0; s < sl->n_slots; ++s) {
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                sl->expert_slot.erase(sl->slot_expert[s]);
                sl->slot_expert[s]  = -1;
                sl->slot_state[s]   = LLAMA_MOE_STREAM_SLOT_EMPTY;
                sl->slot_pending[s] = 0;
                sl->slot_gen[s]++;
            }
        }
    }

    n_inplace_layers = 0;
    released_last    = 0;
    unwire_wait_ms   = 0.0;
    pending_release.clear();
    pending_views.clear();
    pending_views.reserve(cache_bufs.size());
    size_t n_tails = 0;
    for (const auto & group : cache_bufs) {
        for (auto * t = ggml_get_first_tensor(group.ctx.get()); t; t = ggml_get_next_tensor(group.ctx.get(), t)) {
            ++n_tails;
        }
    }
    // View installation must not allocate bookkeeping after changing GPU coverage.
    pending_release.reserve(n_tails);
    const size_t   released0 = released_bytes();
    const uint32_t old_slots = n_slots;
    // An in-place shrink measures the wired pool once the workspace freed just before has settled,
    // so the drop it waits for is the tails' alone.
    const uint64_t wired0 = new_slots < old_slots && can_resize_inplace(new_slots) ? moe_stream_wired_settled(150, 1000) : 0;
    const int64_t  t_alloc0  = stats.t_resize_alloc_us;
    const int64_t  t_copy0   = stats.t_resize_copy_us;
    const int64_t  t_free0   = stats.t_resize_free_us;
    bool ok = true;
    try {
        for (auto & sl : layers) {
            if (sl && !moe_stream_resize_layer(*this, *sl, new_slots)) {
                ok = false;
                break;
            }
            if (sl) {
                migrated.push_back(sl.get());
            }
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: migration exception: %s\n", __func__, err.what());
        ok = false;
    }

    // Do not publish the global capacity until every growing GPU view is usable. Views installed
    // before a failed growth are supersets of the old layout and may safely remain on rollback.
#ifdef GGML_USE_METAL
    if (ok && !pending_views.empty()) {
        const int64_t t_views = ggml_time_us();
        std::atomic<size_t> next{0};
        std::atomic<bool> views_ok{true};
        auto work = [&]() {
            for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < pending_views.size(); ) {
                auto & [group, live, release] = pending_views[i];
                try {
                    if (!moe_stream_apply_views(*this, *group, live, release)) { views_ok = false; }
                } catch (const std::exception & err) {
                    // All throwing preparation precedes view installation; shrink may keep its
                    // old superset, whereas growth must return to its previous dimensions.
                    LLAMA_LOG_WARN("%s: layer %d view preparation failed: %s\n", __func__, group->il, err.what());
                    if (!release) { views_ok = false; }
                }
            }
        };
        std::vector<std::thread> pool;
        try {
            const size_t n_threads = std::min<size_t>(pending_views.size(),
                    std::clamp<unsigned>(std::thread::hardware_concurrency(), 1u, 8u));
            pool.reserve(n_threads - 1);
            for (size_t t = 1; t < n_threads; ++t) { pool.emplace_back(work); }
        } catch (const std::exception &) {
            // Fewer helpers; this thread drains the remaining independent groups.
        }
        work();
        for (auto & th : pool) { th.join(); }
        ok = views_ok.load();
        stats.t_resize_alloc_us += ggml_time_us() - t_views;
    }
#endif

    if (!ok) {
        pending_views.clear();
        pending_release.clear();
        LLAMA_LOG_ERROR("%s: resize failed; restoring the old %u-slot layout\n", __func__, old_slots);
        try {
            for (auto it = migrated.rbegin(); it != migrated.rend(); ++it) {
                if (!moe_stream_resize_layer(*this, **it, old_slots)) {
                    GGML_ABORT("MoE expert cache rollback failed after a partial resize");
                }
            }
        } catch (...) {
            GGML_ABORT("MoE expert cache rollback threw after a partial resize");
        }
        // Rolling back a grow only reduces coverage requirements. Keep any larger installed
        // views instead of performing another fallible re-wrap, and release no pages.
        pending_views.clear();
        pending_release.clear();
    } else {
        n_slots = new_slots;
        pending_views.clear();
        if (!pending_release.empty()) {
            size_t bytes = 0;
            for (const auto & r : pending_release) {
                bytes += r.second;
            }
            const int64_t t_release = ggml_time_us();
            unwire_wait_ms = wait_unwired(wired0, bytes, 3000);
#if defined(GGML_USE_METAL)
            for (const auto & [ptr, size] : pending_release) {
                if (mmap(ptr, size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0) != ptr) {
                    LLAMA_LOG_WARN("%s: failed to release %zu tail bytes; they stay resident\n", __func__, size);
                }
            }
#endif
            pending_release.clear();
            stats.t_resize_free_us += ggml_time_us() - t_release;
        }
        released_last = released_bytes() > released0 ? released_bytes() - released0 : 0;
        LLAMA_LOG_WARN("%s: expert cache %u -> %u slots, %.2f MiB resident allocation; alloc %.1f ms, copy %.1f ms, free %.1f ms (unwire wait %.1f); %zu layers in place\n",
                __func__, old_slots, new_slots, size_bufs()/1024.0/1024.0,
                (stats.t_resize_alloc_us - t_alloc0)/1000.0, (stats.t_resize_copy_us - t_copy0)/1000.0,
                (stats.t_resize_free_us - t_free0)/1000.0, unwire_wait_ms, n_inplace_layers);
    }

    allocation_epoch++; // also invalidate jobs after an attempted migration and rollback

    if (ok && warm_decode && new_slots > old_slots) {
        warm_decode_locked();
    }
    return ok;
}

// dump counter deltas since the last dump, if the interval has elapsed. caller holds mtx.
//
// The number this exists to produce is `stall`: wave-staging wait time as a percentage of wall
// time in the window. Prefill has been measured as neither FLOP-bound (cutting 40% of expert-GEMM
// work made it slower) nor bandwidth-bound (halving per-token expert bytes via -ub changed nothing)
// nor queue-depth-bound (8/12/16 I/O threads are flat), so the open question is whether the time is
// going into waiting on expert loads at all. A high stall% says yes and points at the load path;
// a low one says the cost is elsewhere - the CPU-side staging, the custom op, or the GEMM itself.
void llama_moe_stream::maybe_dump_stats_locked() {
    if (stats_dump_us <= 0) {
        return;
    }

    const int64_t now = ggml_time_us();
    if (stats_t_last_us == 0) {
        stats_t_last_us = now;
        stats_prev      = stats;
        return;
    }
    const int64_t dt = now - stats_t_last_us;
    if (dt < stats_dump_us) {
        return;
    }

    const int64_t d_calls   = stats.n_calls          - stats_prev.n_calls;
    const int64_t d_hit     = stats.n_hit            - stats_prev.n_hit;
    const int64_t d_miss    = stats.n_miss           - stats_prev.n_miss;
    const int64_t d_waves   = stats.n_waves_run      - stats_prev.n_waves_run;
    const int64_t d_pre_i   = stats.n_preload_issued - stats_prev.n_preload_issued;
    const int64_t d_pre_r   = stats.n_preload_ready  - stats_prev.n_preload_ready;
    const int64_t d_hit_l   = stats.n_hit_loading    - stats_prev.n_hit_loading;
    const int64_t d_hit_r   = stats.n_hit_ready      - stats_prev.n_hit_ready;
    const int64_t d_stall   = (stats.t_stall_us      - stats_prev.t_stall_us) +
                              (stats.t_stall_wave_us - stats_prev.t_stall_wave_us);
    const int64_t d_op      = (stats.t_wave_op_us    - stats_prev.t_wave_op_us) +
                              (stats.t_remap_op_us   - stats_prev.t_remap_op_us);
    const int64_t d_lock    = stats.t_remap_lock_us - stats_prev.t_remap_lock_us;
    const int64_t d_cpu     = std::max<int64_t>(0, d_op - d_stall - d_lock);
    const int64_t d_touched = d_hit + d_miss;

    // only report windows that did work, so idle time between requests does not emit noise
    if (d_calls > 0 || d_waves > 0) {
        LLAMA_LOG_WARN("%s: moe stream: %6.2f s | remaps %5" PRId64 " waves %4" PRId64
                       " | miss %5" PRId64 "/%-6" PRId64 " (%5.1f%%) | hit rdy/late %5" PRId64 "/%-5" PRId64
                       " | preload %4" PRId64 "->%-4" PRId64
                       " | stall %5.1f%%  cpu-op %5.1f%%  rest(wall) %5.1f%%  lock %5.1f%%\n",
                __func__, dt/1e6, d_calls, d_waves, d_miss, d_touched,
                d_touched > 0 ? 100.0*d_miss/d_touched : 0.0,
                d_hit_r, d_hit_l,
                d_pre_i, d_pre_r,
                dt > 0 ? 100.0*d_stall/dt      : 0.0,
                dt > 0 ? 100.0*d_cpu/dt        : 0.0,
                dt > 0 ? 100.0*(dt - d_op)/dt  : 0.0,
                dt > 0 ? 100.0*d_lock/dt       : 0.0);

        const int64_t d_self_evict = stats.n_demand_self_evict - stats_prev.n_demand_self_evict;
        const int64_t d_spec = stats.n_spec_dispatch - stats_prev.n_spec_dispatch;
        if (d_spec > 0) {
            LLAMA_LOG_WARN("%s: moe stream: prefetch dispatch %" PRId64 " while-demand %" PRId64
                           " | expert read MiB %.2f\n",
                    __func__, d_spec, stats.n_spec_with_demand - stats_prev.n_spec_with_demand,
                    (stats.n_bytes_read - stats_prev.n_bytes_read)/1048576.0);
        }
        if (d_self_evict > 0) {
            LLAMA_LOG_WARN("%s: moe stream: demand self-evictions = %" PRId64 " in this interval\n", __func__, d_self_evict);
        }

        // per-miss anatomy: is a miss read-bound (parallelise the slab reads) or upload-bound
        // (pipeline read against upload)?
        const int64_t d_rd = stats.t_io_read_us   - stats_prev.t_io_read_us;
        const int64_t d_up = stats.t_io_upload_us - stats_prev.t_io_upload_us;
        const int64_t d_sl = stats.n_slabs_read   - stats_prev.n_slabs_read;
        if (d_sl > 0) {
            LLAMA_LOG_WARN("%s: moe stream: slabs %5" PRId64 " | read %7.2f ms (%5.3f ms/slab) | "
                           "upload %7.2f ms (%5.3f ms/slab) | read %4.1f%% of miss\n",
                    __func__, d_sl, d_rd/1000.0, d_rd/1000.0/d_sl, d_up/1000.0, d_up/1000.0/d_sl,
                    (d_rd + d_up) > 0 ? 100.0*d_rd/(d_rd + d_up) : 0.0);

            // reads faster than the first bound cannot have come from the drive, so they measure
            // the page cache acting as an L2 behind the slot cache
            char hist[256];
            int  off = 0;
            int64_t fast = 0;
            for (int b = 0; b < MOE_STREAM_READ_BUCKETS; b++) {
                const int64_t d = stats.n_read_bucket[b] - stats_prev.n_read_bucket[b];
                if (b == 0) {
                    fast = d;
                }
                off += snprintf(hist + off, sizeof(hist) - off, "%s%" PRId64,
                        b ? "/" : "", d);
            }
            LLAMA_LOG_WARN("%s: moe stream: read us <100/<250/<500/<1k/<2k/<4k/<8k/more = %s | "
                           "page-cache L2 %4.1f%%\n",
                    __func__, hist, d_sl > 0 ? 100.0*fast/d_sl : 0.0);
        }

        if (stats.chunk_util_max > 0) {
            LLAMA_LOG_WARN("%s: moe stream: chunk utilisation worst = %" PRId64 "%% (100%% = abort)\n",
                    __func__, stats.chunk_util_max);
        }

        if (stats.pair_over_max > 0) {
            // running maximum, not a delta: it sizes the graph's chunk slack, so what matters is the
            // worst case the run has produced so far, not the worst in this particular window
            LLAMA_LOG_WARN("%s: moe stream: pair imbalance worst = +%" PRId64 "%% over mean\n",
                    __func__, stats.pair_over_max);
        }
    }

    stats_t_last_us = now;
    stats_prev      = stats;
}

void llama_moe_stream::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    LLAMA_LOG_WARN("%s: moe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_WARN("%s: moe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    LLAMA_LOG_WARN("%s: moe stream: demand self-evictions = %" PRId64 "\n", __func__, stats.n_demand_self_evict);
    LLAMA_LOG_WARN("%s: moe stream: prefetch dispatch %" PRId64 " while-demand %" PRId64
                   " | expert read MiB %.2f\n",
            __func__, stats.n_spec_dispatch, stats.n_spec_with_demand, stats.n_bytes_read/1048576.0);
    if (stats.n_wave_calls > 0) {
        LLAMA_LOG_WARN("%s: moe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\n",
                __func__, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);
    }
    if (stats.n_slabs_read > 0) {
        for (int b = 0; b < MOE_STREAM_READ_BUCKETS; b++) {
            const int64_t lo = b ? MOE_STREAM_READ_BUCKET_US[b-1] : 0;
            if (b == MOE_STREAM_READ_BUCKETS - 1) {
                LLAMA_LOG_WARN("%s: moe stream: read %6" PRId64 " us +      : %8" PRId64 " slabs (%4.1f%%)\n",
                        __func__, lo, stats.n_read_bucket[b], 100.0*stats.n_read_bucket[b]/stats.n_slabs_read);
            } else {
                LLAMA_LOG_WARN("%s: moe stream: read %6" PRId64 " - %6" PRId64 " us: %8" PRId64 " slabs (%4.1f%%)\n",
                        __func__, lo, MOE_STREAM_READ_BUCKET_US[b], stats.n_read_bucket[b],
                        100.0*stats.n_read_bucket[b]/stats.n_slabs_read);
            }
        }
    }
}

// custom-op callback (single-threaded on ith 0): given the router's expert ids, ensure every touched
// expert is resident - reserving cache slots and demand-loading misses, stalling until they commit -
// then rewrite each id to its cache slot. this only relabels ids, so the same experts are computed
// in the same order; the result matches a non-streamed run (bit-exact when both paths use the same
// kernels, as on CUDA; a CPU build that repacks the non-streamed weights can differ in the last bits).
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    // timed from before the lock: lock acquisition is on the critical path too, since the GPU sits
    // idle while this CPU-side op runs as a graph dependency
    const int64_t t_op0 = ggml_time_us();

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    const int64_t t_lock0 = ggml_time_us();
    std::unique_lock<std::mutex> lk(mgr->mtx);
    const int64_t lock_us = ggml_time_us() - t_lock0;

    mgr->maybe_dump_stats_locked();
    mgr->stats.t_remap_lock_us += lock_us;

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly-hot expert ages out
    for (const int32_t e : sl->uniq) {
        sat_inc(sl->route_hotness[e]);
    }
    // Prefill's wave sweep touches every expert, which flattens route_hotness. Decode-width calls
    // alone say what the next decode phase will ask for, so keep them apart for warm_decode_locked.
    if ((uint32_t) a->ne[1] <= 32) {
        for (const int32_t e : sl->uniq) {
            sat_inc(sl->decode_hot[e]);
        }
    }
    if (mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & sl2 : mgr->layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
                for (auto & h : sl2->decode_hot) {
                    h >>= 1;
                }
            }
        }
    }

    // classify the touched experts; reserve and enqueue demand loads in deterministic order
    std::fill(sl->keep.begin(), sl->keep.end(), 0);
    sl->demand_slots.clear();

    bool waited = false;
    for (const int32_t e : sl->uniq) {
        const auto it = sl->expert_slot.find(e);
        if (it != sl->expert_slot.end()) {
            const int32_t s = it->second;
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                mgr->promote_slot_locked(*sl, s);
                waited = true;
                mgr->stats.n_hit_loading++;
                sl->slot_spec[s] = 0; // a prefetch that did not land in time is not a preload hit
            } else {
                mgr->stats.n_hit_ready++;
                if (sl->slot_spec[s]) {
                    mgr->stats.n_preload_ready++; // the lookahead prefetch landed before the demand
                    sl->slot_spec[s] = 0;
                }
            }
            mgr->stats.n_hit++;
            sl->keep[s] = 1;
            sl->demand_slots.push_back(s);
        } else {
            int32_t v;
            while ((v = mgr->pick_victim_locked(*sl, sl->keep.data())) < 0) {
                // every allowed slot is loading; wait for a commit and retry
                const int64_t t_wait = ggml_time_us();
                mgr->cv_done.wait(lk);
                const int64_t waited_us = ggml_time_us() - t_wait;
                mgr->stats.t_victim_wait_us += waited_us;
                mgr->stats.t_stall_us += waited_us;
                if (mgr->load_failed) {
                    GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                }
            }
            if (!sl->seen[e]) {
                mgr->stats.n_miss_cold++;
            }
            const int32_t old = sl->slot_expert[v];
            if (old >= 0 && sl->touched[old]) {
                mgr->stats.n_demand_self_evict++;
            }
            mgr->reserve_slot_locked(*sl, e, v);
            // one work item PER SLAB: the 2-3 slabs of an expert are independent reads, and
            // issuing them together is what lifts device queue depth above 1.
            sl->slot_pending[v] = (uint8_t) sl->weights.size();

            for (size_t wi = 0; wi < sl->weights.size(); wi++) {

                mgr->q_demand.push_back({ sl, e, v, (int32_t) wi, sl->slot_gen[v], mgr->allocation_epoch });
                mgr->cv_work.notify_one();  // one wakeup per slab

            }
            // (workers woken per slab inside the loop above)
            mgr->stats.n_miss++;
            waited = true;
            sl->keep[v] = 1;
            sl->demand_slots.push_back(v);
        }
    }

    // Enqueue current demand first, then predict the next layer while the same lock still keeps
    // readers from taking speculative work ahead of it. The wait releases the lock to both queues.

    if (waited) {
        const int64_t t0 = ggml_time_us();
        mgr->cv_done.wait(lk, [&]{
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t s : sl->demand_slots) {
                if (sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->stats.t_stall_us += ggml_time_us() - t0;
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t s = sl->expert_slot.at(ids[i]);
        sl->slot_last_use[s] = ++sl->use_counter;
        out[i] = s;
    }

    mgr->stats.t_remap_op_us += ggml_time_us() - t_op0;
}

// Prefetch next-layer experts without another graph split. Wrong guesses consume
// read bandwidth and can evict useful residents, so the candidate budget is bounded.
static void llama_moe_stream_prefetch_next(llama_moe_stream_lookahead * la, const float * logits, uint32_t rows) {
    llama_moe_stream_layer & sl = *la->sl_next;
    auto * mgr = sl.mgr;

    const uint32_t n = sl.n_expert;

    llama_moe_lookahead_select(logits, n, rows, la->top_k, la->bias, la->score, la->selected);
    for (size_t rank = 0; rank < la->selected.size(); ++rank) {
        const int32_t best = la->selected[rank];
        if (sl.expert_slot.find((int32_t) best) != sl.expert_slot.end()) {
            continue;                  // already resident or in flight - the common case
        }
        if (mgr->q_spec.size() >= mgr->q_spec_max) {
            break;                     // backlog already deeper than the drive will clear in time
        }
        const int32_t v = mgr->pick_victim_locked(sl, nullptr);
        if (v < 0) {
            break;                     // every slot busy; the layer's own remap will demand-load it
        }
        mgr->reserve_slot_locked(sl, (int32_t) best, v);
        sl.slot_spec[v] = 1;
        sl.slot_pending[v] = (uint8_t) sl.weights.size();
        for (size_t wi = 0; wi < sl.weights.size(); wi++) {
            mgr->q_spec.push_back({ &sl, (int32_t) best, v, (int32_t) wi, sl.slot_gen[v], mgr->allocation_epoch });
            mgr->cv_work.notify_one();
        }
        mgr->stats.n_preload_issued++;
    }
}

void llama_moe_stream_remap_la(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth, void * userdata) {
    auto * la = (llama_moe_stream_lookahead *) userdata;

    if (ith != 0 || la->sl_next == nullptr || la->top_k == 0) {
        llama_moe_stream_remap(dst, a, ith, nth, la->sl);
        return;
    }

    // b holds every row's prediction; llama_moe_lookahead_select uses the last row.
    GGML_ASSERT(b->type == GGML_TYPE_F32 && ggml_is_contiguous(b));
    const int64_t n_tok = b->ne[1] > 0 ? b->ne[1] : 1;
    const float * logits = (const float *) b->data;

    if (!la->bias_read) {
        la->bias_read = true;
        if (la->bias_src) {
            la->bias.resize(ggml_nelements(la->bias_src));
            ggml_backend_tensor_get(la->bias_src, la->bias.data(), 0, ggml_nbytes(la->bias_src));
        }
    }

    llama_moe_stream_remap(dst, a, ith, nth, la->sl);
    std::unique_lock<std::mutex> lk(la->sl_next->mgr->mtx);
    if (!la->sl_next->mgr->load_failed) {
        llama_moe_stream_prefetch_next(la, logits, (uint32_t) n_tok);
    }
}

// stable per-wave userdata; grows lazily and records the per-wave expert capacity (set at build)
llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

bool moe_stream_partition() {
    static const bool v = [] {
        const char * s = std::getenv("LLAMA_MOE_STREAM_PARTITION");
        return s == nullptr || atoi(s) != 0; // on by default; an EMPTY value must mean off, not on
    }();
    return v;
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split them
// into consecutive groups of plan_capacity, one group per wave (sl.expert_wave[e] = e's wave)
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();
    maybe_dump_stats_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (!sl.touched[e]) {
            sl.touched[e] = 1;
            sl.uniq.push_back(e);
        }
    }


    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;

    // Wave slices. The masked path packs uniq into full groups of plan_capacity, which is what it has
    // always done. The partition path CANNOT: the graph sized its pair chunk from the wave count it
    // built, so the planner has to produce exactly that many waves. A ubatch touching few experts
    // otherwise plans fewer, fatter waves than the graph expects, and each then holds more pairs than
    // the chunk allows - a repetitive prompt planned 2 waves against a graph built for 5, putting
    // n_pairs/2 = 1536 pairs into a chunk of 1127. No balancing can fix a wave-count disagreement.
    const size_t n_uniq = sl.uniq.size();
    if (sl.plan_waves_want > 1 && n_uniq >= sl.plan_waves_want) {
        sl.plan_n_waves = sl.plan_waves_want;
    }
    sl.wave_first.assign(sl.plan_n_waves, 0);
    sl.wave_count.assign(sl.plan_n_waves, 0);
    {
        // spread the experts evenly over exactly plan_n_waves slices, never exceeding plan_capacity
        const size_t base = n_uniq/sl.plan_n_waves;
        const size_t rem  = n_uniq%sl.plan_n_waves;
        size_t at = 0;
        for (uint32_t w = 0; w < sl.plan_n_waves; w++) {
            const size_t cnt = std::min<size_t>(base + (w < rem ? 1 : 0), sl.plan_capacity);
            sl.wave_first[w] = (uint32_t) at;
            sl.wave_count[w] = (uint32_t) cnt;
            at += cnt;
        }
        GGML_ASSERT(at == n_uniq); // every touched expert belongs to exactly one wave
        for (uint32_t w = 0; w < sl.plan_n_waves; w++) {
            for (uint32_t i = 0; i < sl.wave_count[w]; i++) {
                sl.expert_wave[sl.uniq[sl.wave_first[w] + i]] = (uint8_t) w;
            }
        }
    }

    // keyed off the chunk the graph set, NOT off the env var: a ubatch too small to partition falls
    // back to the masked path, and planning pairs for it would check against a stale chunk
    if (sl.plan_pair_chunk > 0) {
        plan_pairs_locked(sl, ids, n);
    }
}

// Pair partitioning: give every (token, expert) pair to exactly one wave, so the expert GEMMs cover
// each pair once instead of once per wave. Called after the expert->wave split above, which it
// REORDERS: stage_wave_locked stages uniq[w*cap .. +cap], so keeping the waves as contiguous runs of
// uniq means the staging and preload paths need no changes at all - only the order within uniq moves.
//
// Why reorder: the graph fixes chunk_pairs before the router has run, so an unbalanced split (some
// waves owning far more pairs than others, since routing is skewed) would force chunk_pairs up to the
// worst case and give back the saving. Balancing the pair count across waves bounds it near the mean.
void llama_moe_stream::plan_pairs_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    const uint32_t n_waves = sl.plan_n_waves;
    if (n_waves <= 1) {
        sl.plan_pair.clear();
        return;
    }

    // pairs per expert - the weight each expert contributes to its wave
    sl.pair_count.assign(sl.n_expert, 0);
    for (int64_t i = 0; i < n; i++) {
        sl.pair_count[ids[i]]++;
    }

    // group sizes must match what stage_wave_locked will slice out of uniq: cap for every wave but
    // the last, which takes the remainder
    std::vector<uint32_t> room(sl.wave_count);

    // heaviest expert first into the currently lightest wave that still has room (LPT scheduling):
    // bounds the max wave load near the mean for skewed routing, which is what caps chunk_pairs
    std::vector<int32_t> order(sl.uniq);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return sl.pair_count[a] > sl.pair_count[b];
    });

    // SNAKE order, not greedy-lightest. Each wave must end up with a fixed NUMBER of experts (its
    // staging slice), and that cardinality constraint fights load balance: sending each expert to the
    // lightest wave fills the light waves' expert slots first, after which every remaining expert is
    // forced into whatever wave still has room regardless of its load. With 256 experts over 5 waves
    // of 52 there is almost no spare capacity, so that is forced rather than unlucky - it piled 1667
    // pairs into one wave against a mean of 615, all of them individually small.
    //
    // Sweeping back and forth (rank 0->wave 0, 1->1, .. k-1->k-1, k->k-1, k+1->k-2, ..) pairs each
    // heavy expert with a light one and fills every wave to exactly its room by construction.
    std::vector<std::vector<int32_t>> group(n_waves);
    std::vector<int64_t>              load (n_waves, 0);
    {
        std::vector<uint32_t> left(room);
        std::vector<uint32_t> seq;
        seq.reserve(order.size());

        uint32_t w = 0;
        int      dir = 1;
        while (seq.size() < order.size()) {
            if (left[w] > 0) {
                seq.push_back(w);
                left[w]--;
            }
            if (dir > 0) {
                if (w + 1 < n_waves) { w++; } else { dir = -1; }
            } else {
                if (w > 0) { w--; } else { dir = 1; }
            }
        }
        for (size_t i = 0; i < order.size(); i++) {
            group[seq[i]].push_back(order[i]);
            load [seq[i]] += sl.pair_count[order[i]];
        }
    }

    // Repair pass: swapping a heavy expert out of the worst wave for a lighter one from the best wave
    // preserves both cardinalities, so it can only help. Only runs when a wave is actually over the
    // chunk, which snake ordering already makes rare.
    for (int iter = 0; iter < 64; iter++) {
        uint32_t hi = 0, lo = 0;
        for (uint32_t w = 1; w < n_waves; w++) {
            if (load[w] > load[hi]) { hi = w; }
            if (load[w] < load[lo]) { lo = w; }
        }
        if (load[hi] <= (int64_t) sl.plan_pair_chunk || hi == lo) {
            break;
        }

        // best swap = the one that shrinks the gap most without inverting it
        const int64_t gap = load[hi] - load[lo];
        int64_t best_d = 0;
        size_t  bi = 0, bj = 0;
        for (size_t i = 0; i < group[hi].size(); i++) {
            for (size_t j = 0; j < group[lo].size(); j++) {
                const int64_t d = sl.pair_count[group[hi][i]] - sl.pair_count[group[lo][j]];
                if (d > best_d && 2*d <= gap + best_d) { best_d = d; bi = i; bj = j; }
            }
        }
        if (best_d <= 0) {
            break; // nothing left to trade
        }
        std::swap(group[hi][bi], group[lo][bj]);
        load[hi] -= best_d;
        load[lo] += best_d;
    }

    // rewrite uniq in wave order and record each wave's slice, so the stager needs no change
    sl.uniq.clear();
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.wave_first[w] = (uint32_t) sl.uniq.size();
        sl.wave_count[w] = (uint32_t) group[w].size();
        for (const int32_t e : group[w]) {
            sl.expert_wave[e] = (uint8_t) w;
            sl.uniq.push_back(e);
        }
    }

    // how far the worst wave sits above the mean - what the graph's chunk slack has to cover
    const int64_t mean = (int64_t) n/n_waves;
    for (uint32_t w = 0; w < n_waves; w++) {
        stats.pair_over_max = std::max(stats.pair_over_max, mean > 0 ? (load[w] - mean)*100/mean : 0);
    }

    // CHUNK UTILISATION: the worst wave load as a percentage of the bound that ABORTS when exceeded.
    // This is the number that predicts a crash; imbalance-over-mean does not, because the chunk is
    // floored at n_tokens and so is not a fixed multiple of the mean. 100% means the server died.
    int64_t worst = 0;
    for (uint32_t w = 0; w < n_waves; w++) worst = std::max(worst, load[w]);
    if (sl.plan_pair_chunk > 0) {
        stats.chunk_util_max = std::max(stats.chunk_util_max, worst*100/(int64_t) sl.plan_pair_chunk);
    }

    // flat pair indices (t*n_ids + k) owned by each wave
    sl.plan_pair.assign(n_waves, {});
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.plan_pair[w].reserve((size_t) load[w]);
    }
    for (int64_t i = 0; i < n; i++) {
        sl.plan_pair[sl.expert_wave[ids[i]]].push_back((int32_t) i);
    }

    // ---------------------------------------------------------------------------------------
    // SPLIT PASS: move surplus pairs off any over-full wave, staging that expert in the receiving
    // wave as well.
    //
    // Balancing alone cannot fix this, because all of an expert's pairs go wherever it is staged. A
    // single expert can hold n_tokens pairs (one per token), so even with the chunk floored at
    // n_tokens, that expert PLUS any other in the same wave overflows. Measured: an 800-word
    // repetitive prompt gave "wave 0 holds 1009 pairs but chunk is 1000" - overflowing by the size
    // of the second expert. Repetitive input reaches this trivially, so it is not a corner case.
    //
    // A distribution always exists: total capacity is n_waves*chunk, which exceeds n_pairs by the
    // slack. Only indivisibility stood in the way, and an expert may be staged in more than one wave
    // - it costs a slot there, and a second staging of a resident expert is a cache hit, not I/O.
    // ---------------------------------------------------------------------------------------------
    const size_t chunk = sl.plan_pair_chunk;
    for (uint32_t w = 0; w < n_waves && chunk > 0; w++) {
        while (sl.plan_pair[w].size() > chunk) {
            const size_t surplus = sl.plan_pair[w].size() - chunk;

            // the expert contributing most to this wave is the one worth moving
            std::unordered_map<int32_t, size_t> cnt;
            for (const int32_t idx : sl.plan_pair[w]) cnt[ids[idx]]++;
            int32_t hot = -1; size_t hot_n = 0;
            for (const auto & kv : cnt) if (kv.second > hot_n) { hot = kv.first; hot_n = kv.second; }
            if (hot < 0) break;

            // A receiving wave always needs pair room. It needs a free expert slot only if it does
            // not already stage `hot` - if it does, the pairs land on a slot that wave has already
            // reserved, so the move is free. Preferring those receivers is what the earlier version
            // had backwards: it SKIPPED them, spending a slot where none was needed.
            int32_t dst = -1; size_t room = 0; bool dst_has = false;
            for (uint32_t v = 0; v < n_waves; v++) {
                if (v == w || sl.plan_pair[v].size() >= chunk) continue;

                const bool has = std::find(group[v].begin(), group[v].end(), hot) != group[v].end();
                if (!has && group[v].size() >= sl.plan_capacity) continue;

                // a free move beats a bigger one that costs a slot
                const size_t r = chunk - sl.plan_pair[v].size();
                if (has != dst_has ? has : r > room) { room = r; dst = (int32_t) v; dst_has = has; }
            }
            if (dst < 0) {
                break; // no receiver; the check below reports it rather than truncating silently
            }

            const size_t move = std::min({surplus, room, hot_n});
            std::vector<int32_t> keep; keep.reserve(sl.plan_pair[w].size() - move);
            size_t moved = 0;
            for (const int32_t idx : sl.plan_pair[w]) {
                if (moved < move && ids[idx] == hot) { sl.plan_pair[dst].push_back(idx); moved++; }
                else                                  { keep.push_back(idx); }
            }
            sl.plan_pair[w].swap(keep);
            if (!dst_has) {
                group[dst].push_back(hot);   // stage it in the receiver too
            }
            stats.n_pair_splits++;
            if (moved == 0) break;
        }
    }

    // uniq must reflect the split staging, so rebuild the slices from the (possibly grown) groups.
    // expert_wave is left as the last writer sets it: the partition path keys off plan_pair, not
    // expert_wave, and the masked path never runs when partitioning is active.
    sl.uniq.clear();
    for (uint32_t w = 0; w < n_waves; w++) {
        sl.wave_first[w] = (uint32_t) sl.uniq.size();
        sl.wave_count[w] = (uint32_t) group[w].size();
        for (const int32_t e : group[w]) sl.uniq.push_back(e);
    }

    // still loud if a wave cannot be represented - but this should now be unreachable
    for (uint32_t w = 0; w < n_waves; w++) {
        if (sl.plan_pair[w].size() > sl.plan_pair_chunk) {
            // Report which constraint bound, because the two have different fixes: no pair room
            // means the chunk itself is too small (raise LLAMA_MOE_STREAM_PAIR_SLACK), no slot room
            // means the wave count is too low for the expert count (the cap-1 sizing in
            // llama-graph.cpp did not apply, e.g. the pairs-per-wave floor blocked it).
            size_t free_pairs = 0, free_slots = 0;
            for (uint32_t v = 0; v < n_waves; v++) {
                free_pairs += sl.plan_pair[v].size()   < chunk            ? 1 : 0;
                free_slots += group[v].size() < (size_t) sl.plan_capacity ? 1 : 0;
            }
            GGML_ABORT("MoE expert streaming: wave %u holds %zu pairs but chunk is %u after splitting "
                       "(%u waves, %zu with pair room, %zu with a free expert slot, capacity %u, "
                       "%zu experts staged); report it with the prompt that caused it",
                    w, sl.plan_pair[w].size(), sl.plan_pair_chunk,
                    n_waves, free_pairs, free_slots, sl.plan_capacity, sl.uniq.size());
        }
    }
}

// make wave w's expert slice (uniq[wave_first[w] .. +wave_count[w])) resident, waiting for its loads,
// and best-effort preload the next wave so its loads overlap this wave's compute. leaves
// sl.demand_slots = this wave's slots and sl.plan_pool = the resident parking pool (>= n_ids slots)
// the emit draws masked pairs from
void llama_moe_stream::stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids) {
    // the slices come from plan_waves_locked rather than being w*plan_capacity: the partition path
    // needs exactly as many waves as the graph built, which may be more than uniq/plan_capacity
    const size_t first = (size_t) w < sl.wave_first.size() ? sl.wave_first[w] : sl.uniq.size();
    const size_t count = (size_t) w < sl.wave_count.size() ? sl.wave_count[w] : 0;

    std::fill(sl.keep.begin(), sl.keep.end(), 0);
    sl.demand_slots.clear();

    // a small final wave has fewer than n_ids own slots; borrow the rest from the previous wave's
    //   pool so every token row has n_ids distinct resident parking slots for its masked pairs
    std::vector<int32_t> borrowed;
    if (count < n_ids) {
        GGML_ASSERT(sl.plan_pool.size() >= n_ids - count);
        for (size_t i = 0; i < n_ids - count; i++) {
            borrowed.push_back(sl.plan_pool[i]);
            sl.keep[sl.plan_pool[i]] = 1; // parking slots must survive this wave's loads
        }
    }

    // protect the next wave's already-resident experts so this wave's victims do not evict them.
    //
    // Bounded by the slack actually available. Upstream relies on an implicit invariant - capacity is
    // (n_slots - n_expert_used)/2, so this wave's cap slots + the next wave's cap slots + n_expert_used
    // parking slots exactly fit - and protecting an unbounded next wave is only safe because of it.
    // Raise the capacity without this bound and the keep-set can cover every slot, at which point
    // pick_victim_locked returns -1 forever and the demand loop below blocks on a cv_done nobody will
    // signal: a silent hang at ~0.1% CPU. Deriving the limit from the slack instead makes any capacity
    // safe, and leaves the /2 case behaving exactly as before (slack there is >= cap).
    const size_t nw     = (size_t) w + 1;
    const size_t nfirst = nw < sl.wave_first.size() ? sl.wave_first[nw] : sl.uniq.size();
    const size_t ncount = nw < sl.wave_count.size() ? sl.wave_count[nw] : 0;

    const size_t reserved  = (size_t) sl.plan_capacity + n_ids + 1; // this wave, parking, one victim
    const size_t max_keep  = sl.n_slots > reserved ? sl.n_slots - reserved : 0;

    size_t n_kept = 0;
    for (size_t i = nfirst; i < nfirst + ncount && n_kept < max_keep; i++) {
        const auto it = sl.expert_slot.find(sl.uniq[i]);
        if (it != sl.expert_slot.end()) {
            sl.keep[it->second] = 1;
            n_kept++;
        }
    }

    // reserve and demand-load this wave's experts (per-expert, same path as the decode remap)
    bool waited = false;
    if (count > 0) {
        stats.n_waves_run++;
        for (size_t i = first; i < first + count; i++) {
            const int32_t e  = sl.uniq[i];
            const auto    it = sl.expert_slot.find(e);
            if (it != sl.expert_slot.end()) {
                // already in the cache (resident, or still loading from the previous wave's preload)
                const int32_t s = it->second;
                if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                    promote_slot_locked(sl, s);
                    waited = true;
                } else {
                    stats.n_preload_ready++; // resident from the previous wave's preload
                }
                stats.n_hit++;
                sl.keep[s] = 1;
                sl.demand_slots.push_back(s);
            } else {
                // miss: evict a non-kept slot and queue the load
                int32_t v;
                while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
                    if (load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                }
                if (!sl.seen[e]) {
                    stats.n_miss_cold++;
                }
                reserve_slot_locked(sl, e, v);
                // one work item PER SLAB: the 2-3 slabs of an expert are independent reads, and
                // issuing them together is what lifts device queue depth above 1.
                sl.slot_pending[v] = (uint8_t) sl.weights.size();

                for (size_t wi = 0; wi < sl.weights.size(); wi++) {

                    q_demand.push_back({ &sl, e, v, (int32_t) wi, sl.slot_gen[v], allocation_epoch });
                    cv_work.notify_one();  // one wakeup per slab

                }
                // (workers woken per slab inside the loop above)
                stats.n_miss++;
                waited = true;
                sl.keep[v] = 1;
                sl.demand_slots.push_back(v);
            }
        }
    }

    // best-effort preload of the next wave so its loads overlap this wave's compute; never waits,
    //   whatever cannot be reserved now simply becomes the next wave's demand load
    if (std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const int32_t e = sl.uniq[i];
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, sl.keep.data());
            if (v < 0) {
                continue;
            }
            if (!sl.seen[e]) {
                stats.n_miss_cold++;
            }
            reserve_slot_locked(sl, e, v);
            sl.keep[v] = 1;
            // one work item PER SLAB: the 2-3 slabs of an expert are independent reads, and
            // issuing them together is what lifts device queue depth above 1.
            sl.slot_pending[v] = (uint8_t) sl.weights.size();

            for (size_t wi = 0; wi < sl.weights.size(); wi++) {

                q_demand.push_back({ &sl, e, v, (int32_t) wi, sl.slot_gen[v], allocation_epoch });
                cv_work.notify_one();  // one wakeup per slab

            }
            // (workers woken per slab inside the loop above)
            stats.n_preload_issued++;
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        cv_done.wait(lk, [&]{
            if (load_failed) {
                return true;
            }
            for (const int32_t s : sl.demand_slots) {
                if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        stats.t_stall_wave_us += ggml_time_us() - t0;
    }

    // parking pool: this wave's own resident slots plus the borrowed ones (all keep-protected;
    //   the next same-layer reservation is ordered after this wave's GEMMs by the graph)
    sl.plan_pool = sl.demand_slots;
    sl.plan_pool.insert(sl.plan_pool.end(), borrowed.begin(), borrowed.end());
    GGML_ASSERT(sl.plan_pool.size() >= n_ids);
}

// write out[i] = the cache slot the GEMM should index for each (token, expert) pair of wave w, one
// token row at a time: pairs whose expert is in this wave get its real slot; the rest park on distinct
// resident pool slots (pool_used prevents a repeat within the row, required by the Metal kernel)
void llama_moe_stream::emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_tok) {
    for (int64_t t = 0; t < n_tok; t++) {
        sl.pool_used.clear();

        // pass 1: pairs whose expert belongs to this wave -> that expert's real (resident) slot
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            const int32_t e = ids[i];
            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] == (uint8_t) w) {
                const int32_t s = sl.expert_slot.at(e);
                GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                sl.slot_last_use[s] = ++sl.use_counter;
                out[i] = s;
                sl.pool_used.push_back(s);
            }
        }

        // pass 2: the remaining (masked) pairs -> the next pool slot not yet used in this row
        size_t pi = 0;
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            if (sl.expert_wave[ids[i]] == (uint8_t) w) {
                continue;
            }
            while (std::find(sl.pool_used.begin(), sl.pool_used.end(), sl.plan_pool[pi]) != sl.pool_used.end()) {
                pi++;
                GGML_ASSERT(pi < sl.plan_pool.size());
            }
            GGML_ASSERT(sl.slot_state[sl.plan_pool[pi]] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            out[i] = sl.plan_pool[pi];
            sl.pool_used.push_back(sl.plan_pool[pi]);
            pi++;
        }
    }
}

// Padding may carry slot -1 when every expert cache is a Metal buffer: the Metal mul_mm_id map0 drops an
// id that matches no slot, so the GEMMs spend no tiles on it and leave its rows unwritten, and those rows
// scatter to the scratch row. Other backends index weights by the id directly, so they keep the repeats.
bool llama_moe_stream::pad_skip_ok() {
    if (pad_skip < 0) {
        const char * e = getenv("LLAMA_MOE_STREAM_PAD_SKIP");
        bool ok = !(e && atoi(e) == 0);
        bool any = false;
        for (const auto & group : cache_bufs) {
            if (!group.buf) {
                continue;
            }
            any = true;
#ifdef GGML_USE_METAL
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(group.buf.get()));
            ok = ok && dev != nullptr && ggml_backend_dev_backend_reg(dev) == ggml_backend_metal_reg();
#else
            ok = false;
#endif
        }
        pad_skip = ok && any ? 1 : 0;
    }
    return pad_skip > 0;
}

// Partition path: write the index rows describing wave w's dense pair list. Every (token, expert)
// pair belongs to exactly one wave, so across the waves each pair is emitted once and the GEMM covers
// it once - as opposed to emit_wave_slots above, where every wave covers every pair and masks the rest.
//
// The list is padded to the static chunk by REPEATING this wave's last pair: the GEMM then recomputes
// that pair and the scatter writes the same value to the same row, so padding needs neither a scratch
// row nor zero-initialised output.
// On Metal the padding instead carries slot -1 and goes to the scratch row (see pad_skip_ok).
//
// A wave can also own nothing at all - the graph fixes the wave count from the worst case (every expert
// touched), so a ubatch that touches fewer leaves the late waves empty. Such a wave has no pair it may
// legitimately write, and cannot borrow one either: another wave's expert is not necessarily still
// resident by the time this one runs. It therefore computes a throwaway row on a parked slot and
// scatters it to the scratch row past the end of the real pairs, which nothing reads.
void llama_moe_stream::emit_wave_pairs(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_pairs, int64_t chunk) {
    const std::vector<int32_t> * pairs = (size_t) w < sl.plan_pair.size() ? &sl.plan_pair[w] : nullptr;
    if (pairs != nullptr && pairs->empty()) {
        pairs = nullptr;
    }
    GGML_ASSERT(pairs == nullptr || (int64_t) pairs->size() <= chunk);

    int32_t * r_tok  = out + LLAMA_MOE_PAIR_TOK *chunk;
    int32_t * r_pair = out + LLAMA_MOE_PAIR_PAIR*chunk;
    int32_t * r_slot = out + LLAMA_MOE_PAIR_SLOT*chunk;
    int32_t * r_exp  = out + LLAMA_MOE_PAIR_EXP *chunk;
    int32_t * r_input = out + LLAMA_MOE_PAIR_INPUT*chunk;

    // 32: below it Metal takes the mul_mv_id path, which reads the slot id without a check
    const bool skip = chunk >= 32 && pad_skip_ok();

    if (pairs == nullptr) {
        GGML_ASSERT(!sl.plan_pool.empty()); // stage_wave_locked leaves >= n_ids resident parking slots
        const int32_t s = sl.plan_pool[0];
        GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
        for (int64_t p = 0; p < chunk; p++) {
            r_tok [p] = 0;
            r_pair[p] = (int32_t) n_pairs; // scratch row
            r_input[p] = 0;
            r_slot[p] = skip ? -1 : s;
            r_exp [p] = sl.slot_expert[s];
        }
        return;
    }

    for (int64_t p = 0; p < chunk; p++) {
        const bool pad = p >= (int64_t) pairs->size();
        const int32_t i = (*pairs)[pad ? pairs->size() - 1 : (size_t) p];
        const int32_t e = ids[i];
        const int32_t s = sl.expert_slot.at(e);
        GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
        sl.slot_last_use[s] = ++sl.use_counter;

        r_tok [p] = i/(int32_t) n_ids;
        r_pair[p] = pad && skip ? (int32_t) n_pairs : i;
        r_input[p] = i;
        r_slot[p] = pad && skip ? -1 : s;
        r_exp [p] = e;
    }
}

// Shared preamble of the two wave ops: plan the whole ubatch on wave 0, enforce that the waves run in
// order, and make wave w's expert slice resident while preloading the next. Returns with the manager
// mutex still held, because the emit that follows reads the slot table this just settled.
static std::unique_lock<std::mutex> stage_wave_for_op(llama_moe_stream_layer & sl, int32_t w,
        const int32_t * ids, int64_t n, uint32_t n_ids) {
    auto * mgr = sl.mgr;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_wave_calls++;

    if (w == 0) {
        mgr->plan_waves_locked(sl, ids, n);
    }
    GGML_ASSERT(sl.plan_next_wave == w); // waves must run in order (enforced by the graph ordering token)

    mgr->stage_wave_locked(lk, sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl.plan_next_wave = w + 1;

    return lk;
}

// Custom-op callback for one pass of multi-pass prefill. When a ubatch touches more experts than the
// cache holds, build_moe_ffn runs the expert GEMMs in several waves; this runs once per wave (single-
// threaded on ith 0), in wave order. For wave w it makes that wave's expert slice resident (preloading
// the next wave), then writes the slot ids the GEMM indexes - see plan_waves_locked / stage_wave_locked
// / emit_wave_slots. The router's expert choice is untouched, so the output matches a non-streamed run.
void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    // timed from before the lock: lock acquisition is on the critical path too, since the GPU sits
    // idle while this CPU-side op runs as a graph dependency
    const int64_t t_op0 = ggml_time_us();

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data); // the emit must not clobber the ids other waves read

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk = stage_wave_for_op(*sl, w, ids, n, (uint32_t) a->ne[0]);

    mgr->emit_wave_slots(*sl, ids, out, w, (uint32_t) a->ne[0], a->ne[1]);

    mgr->stats.t_wave_op_us += ggml_time_us() - t_op0;
}

// Partition path (LLAMA_MOE_STREAM_PARTITION=1) counterpart of llama_moe_stream_wave_ids: identical
// staging, but instead of slot ids for every pair it emits the index rows of wave w's own pairs, which
// build_moe_ffn gathers into a dense GEMM and scatters back. No mask op is needed - a pair is computed
// by the one wave that owns it, so there is nothing to discard.
void llama_moe_stream_wave_pairs(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    const int64_t t_op0 = ggml_time_us();

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(dst->ne[1] == LLAMA_MOE_PAIR_ROWS);

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk = stage_wave_for_op(*sl, w, ids, n, (uint32_t) a->ne[0]);

    mgr->emit_wave_pairs(*sl, ids, out, w, (uint32_t) a->ne[0], n, dst->ne[0]);

    mgr->stats.t_wave_op_us += ggml_time_us() - t_op0;
}

// multi-pass prefill: 1.0 for pairs whose expert belongs to wave w, 0.0 otherwise; multiplied into
// this wave's expert GEMM output so the masked-out (parked) pairs contribute nothing to the sum
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          float   * out = (float *) dst->data;

    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run

    for (int64_t i = 0; i < n; i++) {
        out[i] = sl->expert_wave[ids[i]] == (uint8_t) w ? 1.0f : 0.0f;
    }
}
