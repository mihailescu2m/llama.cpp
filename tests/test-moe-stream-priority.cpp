// Real worker queues and file reads, with one CPU-buffer upload held at a controlled boundary.
// This exposes outstanding reads after the demand queue is empty without timing disk latency.
#include "../src/llama-moe-stream.h"
#include "../src/llama-moe-lookahead.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)
using namespace std::chrono_literals;

struct upload_gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool released = false;
    void (*original)(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) = nullptr;

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};

// Each fixture finishes and joins its workers before the next fixture installs a gate.
static upload_gate * active_gate = nullptr;

static void gated_upload(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                         const void * data, size_t offset, size_t size) {
    auto & gate = *active_gate;
    if (offset == 0) {
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.entered = true;
        gate.cv.notify_all();
        gate.cv.wait(lock, [&] { return gate.released; });
    }
    gate.original(buffer, tensor, data, offset, size);
}

struct fixture {
    upload_gate gate;
    std::unique_ptr<llama_moe_stream> stream;
    FILE * file = nullptr;

    fixture(uint32_t slots = 4) : stream(new llama_moe_stream(1, slots, 2, false)) {
        stream->no_zerocopy = true;
        ggml_context_ptr meta{ggml_init({ggml_tensor_overhead()*2, nullptr, true})};
        auto * wt = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, 32, 1, 16);
        stream->create_cache_tensor(0, ggml_backend_cpu_buffer_type(), wt, 0, 0);
        stream->alloc_bufs(false);
        auto * buffer = stream->cache_bufs.front().buf.get();
        gate.original = buffer->iface.set_tensor;
        buffer->iface.set_tensor = gated_upload;
        active_gate = &gate;
        file = tmpfile(); CHECK(file);
        std::vector<float> bytes(32*16, 0.25f);
        CHECK(fwrite(bytes.data(), sizeof(float), bytes.size(), file) == bytes.size());
        CHECK(fflush(file) == 0);
        stream->files.emplace_back(new llama_file(file));
    }

    ~fixture() {
        gate.release(); // also guarantees failure paths cannot leave a joined worker blocked
        stream.reset();
        if (file) { fclose(file); }
        active_gate = nullptr;
    }

    void enqueue_locked(int slot, bool demand) {
        auto & sl = *stream->layers[0];
        stream->reserve_slot_locked(sl, slot, slot);
        sl.slot_pending[slot] = 1;
        auto & queue = demand ? stream->q_demand : stream->q_spec;
        queue.push_back({&sl, slot, slot, 0, sl.slot_gen[slot], stream->allocation_epoch});
        stream->cv_work.notify_all();
    }

    void start_blocked(bool demand) {
        {
            std::lock_guard<std::mutex> lock(stream->mtx);
            enqueue_locked(0, demand);
            stream->start_workers_locked();
        }
        std::unique_lock<std::mutex> lock(gate.mutex);
        CHECK(gate.cv.wait_for(lock, 3s, [&] { return gate.entered; }));
    }

    // `line` names the caller, so a timeout says which wait never completed and in what state
    template<class Predicate> void await(Predicate predicate, int line) {
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        std::unique_lock<std::mutex> lock(stream->mtx);
        while (!predicate()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                const auto & sl = *stream->layers[0];
                char msg[512];
                snprintf(msg, sizeof(msg), "await at line %d timed out: q_demand %zu q_spec %zu workers_active %zu "
                         "demand_inflight %zu ple_pending %zu spec_dispatch %lld slot states %d %d %d %d",
                         line, stream->q_demand.size(), stream->q_spec.size(), (size_t) stream->workers_active,
                         stream->demand_inflight, stream->ple_pending,
                         (long long) stream->stats.n_spec_dispatch, sl.slot_state[0],
                         sl.n_slots > 1 ? sl.slot_state[1] : -1,
                         sl.n_slots > 2 ? sl.slot_state[2] : -1,
                         sl.n_slots > 3 ? sl.slot_state[3] : -1);
                throw std::runtime_error(msg);
            }
            stream->cv_done.wait_for(lock, 1ms);
        }
    }

    void drained() {
        await([&] { return stream->workers_active == 0 && stream->q_demand.empty() && stream->q_spec.empty(); }, __LINE__);
        std::lock_guard<std::mutex> lock(stream->mtx);
        CHECK(stream->demand_inflight == 0);
        for (const auto & w : stream->reads_inflight) { CHECK(w.sl == nullptr); }
    }
};

static void test_victim_wait_accounting() {
    fixture f(1);
    f.start_blocked(false); // the only slot is loading an expert we will not request
    ggml_context_ptr ctx{ggml_init({4096, nullptr, false})};
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    auto * out = ggml_dup_tensor(ctx.get(), ids);
    ((int32_t *) ids->data)[0] = 1;
    std::thread demand([&] {
        llama_moe_stream_remap(out, ids, 0, 1, f.stream->layers[0].get());
    });
    try {
        // The callback retains the manager lock until cv_done.wait. Observing touched
        // under that lock proves it reached the blocked victim-selection path.
        f.await([&] {
            const auto & touched = f.stream->layers[0]->touched;
            return touched.size() > 1 && touched[1] != 0;
        }, __LINE__);
        std::this_thread::sleep_for(5ms);
    } catch (...) {
        f.gate.release(); demand.join(); throw;
    }
    f.gate.release();
    demand.join();
    f.drained();
    CHECK(((int32_t *) out->data)[0] == 0);
    CHECK(f.stream->layers[0]->slot_expert[0] == 1);
    CHECK(f.stream->stats.t_victim_wait_us > 0);
    CHECK(f.stream->stats.t_stall_us >= f.stream->stats.t_victim_wait_us);
    CHECK(f.stream->stats.t_remap_op_us >= f.stream->stats.t_stall_us);
}

static void test_completion() {
    fixture f;
    f.start_blocked(true);
    {
        std::lock_guard<std::mutex> lock(f.stream->mtx);
        CHECK(f.stream->q_demand.empty() && f.stream->demand_inflight == 1);
        f.enqueue_locked(1, false);
    }
    // a speculative read runs beside the outstanding demand read
    f.await([&] { return f.stream->layers[0]->slot_state[1] == LLAMA_MOE_STREAM_SLOT_RESIDENT; }, __LINE__);
    {
        std::lock_guard<std::mutex> lock(f.stream->mtx);
        CHECK(f.stream->layers[0]->slot_state[0] == LLAMA_MOE_STREAM_SLOT_LOADING);
        CHECK(f.stream->stats.n_spec_with_demand == 1);
    }
    f.gate.release();
    f.drained();
    CHECK(f.stream->layers[0]->slot_state[1] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
    CHECK(f.stream->stats.n_slabs_read == 2);
    CHECK(f.stream->stats.n_bytes_read == 2*32*sizeof(float));
    CHECK(f.stream->resize_slots(6)); // a held read must not strand a subsequent allocation transition
}

static void test_promotion() {
    fixture f;
    f.start_blocked(false);
    {
        std::lock_guard<std::mutex> lock(f.stream->mtx);
        auto & sl = *f.stream->layers[0];
        CHECK(f.stream->demand_inflight == 0);
        f.stream->promote_slot_locked(sl, 0);
        f.stream->promote_slot_locked(sl, 0); // repeated promotion must not double-count
        CHECK(f.stream->demand_inflight == 1);
        f.enqueue_locked(1, false);
    }
    f.await([&] { return f.stream->layers[0]->slot_state[1] == LLAMA_MOE_STREAM_SLOT_RESIDENT; }, __LINE__);
    {
        std::lock_guard<std::mutex> lock(f.stream->mtx);
        CHECK(f.stream->demand_inflight == 1);
    }
    f.gate.release();
    f.drained();
    CHECK(f.stream->stats.n_slabs_read == 2); // no duplicated/reissued slabs on promotion
    CHECK(f.stream->stats.n_spec_with_demand == 1);
}

static void test_shutdown() {
    fixture f;
    f.start_blocked(true);
    {
        std::lock_guard<std::mutex> lock(f.stream->mtx);
        f.enqueue_locked(2, false);
    }
    f.await([&] { return f.stream->layers[0]->slot_state[2] == LLAMA_MOE_STREAM_SLOT_RESIDENT; }, __LINE__);
    // Fixture destruction releases the blocked worker and shuts down the idle one.
}

static void test_failed_and_stale_reads() {
    fixture f;
    {
        std::lock_guard<std::mutex> lock(f.stream->mtx);
        f.stream->q_demand.push_back({f.stream->layers[0].get(), 0, 999, 0, 0, 0});
        // Force pread to hit EOF, with a speculative read beside it.
        f.stream->layers[0]->weights[0].offs = 1u << 20;
        f.enqueue_locked(1, true);
        f.enqueue_locked(2, false);
        f.stream->start_workers_locked();
    }
    f.drained();
    CHECK(f.stream->load_failed);
    CHECK(f.stream->stats.n_slabs_read == 2 && f.stream->stats.n_bytes_read == 0);
}

static void test_lookahead_select() {
    const float logits[] = {9, 8, 1, 0,  1, 0, 9, 8};
    std::vector<float> scratch;
    std::vector<int32_t> picked;
    llama_moe_lookahead_select(logits, 4, 2, 2, {}, scratch, picked); // the last row predicts
    CHECK((picked == std::vector<int32_t>{2, 3}));
    llama_moe_lookahead_select(logits, 4, 1, 2, {0, 0, 10, 0}, scratch, picked);
    CHECK((picked == std::vector<int32_t>{2, 0}));
    const float invalid[] = {1, 0, 9, 8,  NAN, -INFINITY, INFINITY, NAN};
    llama_moe_lookahead_select(invalid, 4, 2, 4, {}, scratch, picked); // nonfinite scores never qualify
    CHECK((picked == std::vector<int32_t>{1}));
    llama_moe_lookahead_select(logits, 4, 2, 0, {}, scratch, picked);
    CHECK(picked.empty());
    llama_moe_lookahead_select(logits, 4, 2, 99, {}, scratch, picked); // K clamps to the expert count
    CHECK(picked.size() == 4);
}

int main() {
    try {
        test_lookahead_select();
        test_victim_wait_accounting();
        test_completion();
        test_promotion();
        test_shutdown();
        test_failed_and_stale_reads();
        puts("MoE stream priority: PASS");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
