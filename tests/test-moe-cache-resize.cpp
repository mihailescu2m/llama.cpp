// CPU-only regression coverage: real backend buffers, mixed expert slab sizes, allocation
// failures, queued I/O and whole-graph execution after rebinding stable weight tensors.
#include "../src/llama-moe-stream.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

struct allocation_probe {
    int fail_at = -1;
    bool throws = false;
    llama_moe_stream * stream = nullptr;
    size_t peak = 0;
};

static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t buft, size_t size) {
    auto & probe = *static_cast<allocation_probe *>(buft->context);
    if (probe.fail_at >= 0 && probe.fail_at-- == 0) {
        if (probe.throws) { throw std::bad_alloc(); }
        return nullptr;
    }
    probe.peak = std::max(probe.peak, probe.stream->size_bufs() + size);
    return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
}

struct fixture {
    llama_moe_stream stream{3, 4, 2, false};
    allocation_probe probe;
    ggml_backend_buffer_type buft = *ggml_backend_cpu_buffer_type();
    ggml_context_ptr meta{ggml_init({ggml_tensor_overhead()*16, nullptr, true})};
    uint32_t cap_slots = 0; // CPU buffers resized in place keep their larger allocation

    fixture() {
        probe.stream = &stream;
        buft.context = &probe;
        buft.iface.alloc_buffer = allocate;
        for (int il = 0; il < 3; ++il) {
            for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_MXFP4}) {
                auto * t = ggml_new_tensor_3d(meta.get(), type, 32, il + 1, 16);
                ggml_format_name(t, "blk.%d.weight.%d", il, (int) type);
                stream.create_cache_tensor(il, &buft, t, 0, 0);
            }
        }
        stream.alloc_bufs(false);
    }

    void fill() {
        for (auto & sl : stream.layers) {
            for (uint32_t s = 0; s < sl->n_slots; ++s) {
                sl->slot_expert[s] = s;
                sl->slot_state[s] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
                sl->expert_slot[s] = s;
                sl->slot_last_use[s] = s;
                sl->route_hotness[s] = 10 - s;
                for (const auto & wt : sl->weights) {
                    std::vector<uint8_t> bytes(wt.nb_expert, uint8_t(1 + s + sl->il * 16));
                    ggml_backend_tensor_set(wt.cache, bytes.data(), s*wt.nb_expert, bytes.size());
                }
            }
        }
    }

    void verify() {
        for (const auto & sl : stream.layers) {
            CHECK(sl->n_slots == stream.n_slots);
            CHECK(sl->keep.size() == sl->n_slots);
            CHECK(sl->slot_spec.size() == sl->n_slots);
            CHECK(std::all_of(sl->slot_spec.begin(), sl->slot_spec.end(), [](uint8_t s) { return s == 0; }));
            size_t n_resident = 0;
            for (uint32_t s = 0; s < sl->n_slots; ++s) {
                if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
                    CHECK(sl->slot_expert[s] == -1);
                    continue;
                }
                CHECK(sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                CHECK(sl->slot_pending[s] == 0);
                const int expert = sl->slot_expert[s];
                CHECK(sl->expert_slot.at(expert) == (int) s);
                n_resident++;
                for (const auto & wt : sl->weights) {
                    CHECK(wt.cache->ne[2] == sl->n_slots);
                    CHECK(ggml_is_contiguous(wt.cache));
                    std::vector<uint8_t> bytes(wt.nb_expert);
                    ggml_backend_tensor_get(wt.cache, bytes.data(), s*wt.nb_expert, bytes.size());
                    CHECK(std::all_of(bytes.begin(), bytes.end(), [&](uint8_t b) {
                        return b == uint8_t(1 + expert + sl->il*16);
                    }));
                }
            }
            CHECK(n_resident == sl->expert_slot.size());
        }
        CHECK(stream.size_bufs() == stream.allocation_size(cap_slots ? cap_slots : stream.n_slots));
        CHECK(!stream.resizing);
    }
};

static void test_policy() {
    fixture f;
    const uint64_t base = f.stream.allocation_size(4);
    CHECK(f.stream.slots_for_reclaimed(0, 512) == 4);
    CHECK(f.stream.slots_for_reclaimed(511, 512) == 4);
    for (uint64_t available = 0; available < 150000; available += 211) {
        uint32_t expected = 4;
        for (uint32_t n = 5; n < 16; ++n) {
            uint64_t scratch = 0;
            uint64_t bytes = f.stream.allocation_size(n, &scratch);
            if (bytes - base + scratch + 512 <= available) { expected = n; }
        }
        CHECK(f.stream.slots_for_reclaimed(available, 512) == expected);
    }
    CHECK(f.stream.slots_for_budget(f.stream.allocation_size(8), 4) == 8);
    CHECK(f.stream.slots_for_budget(f.stream.allocation_size(8) - 1, 4) == 7);
}

static void test_residency_and_failures() {
    for (bool populated : {false, true}) {
        fixture f;
        if (populated) { f.fill(); }
        const auto * stable = f.stream.layers[0]->weights[0].cache;
        for (int i = 0; i < 5; ++i) {
            CHECK(f.stream.resize_slots(10)); f.verify();
            CHECK(f.stream.resize_slots(4)); f.verify();
            CHECK(f.stream.layers[0]->weights[0].cache == stable);
        }
        uint64_t scratch = 0;
        uint64_t total = f.stream.allocation_size(10, &scratch);
        CHECK(f.probe.peak <= total + scratch);
    }
    // Allocation failure at every layer, including exceptions, must restore a uniform layout.
    for (bool throws : {false, true}) {
        for (int layer = 0; layer < 3; ++layer) {
            fixture f; f.fill();
            f.probe.fail_at = layer; f.probe.throws = throws;
            CHECK(!f.stream.resize_slots(10)); f.verify();
            CHECK(f.stream.n_slots == 4);
            CHECK(f.stream.resize_slots(10)); f.verify();
            f.probe.fail_at = layer;
            CHECK(!f.stream.resize_slots(2)); f.verify();
            CHECK(f.stream.n_slots == 10);
        }
    }
    {
        fixture f; f.fill();
        CHECK(f.stream.resize_slots(2)); f.verify();
        for (const auto & sl : f.stream.layers) {
            CHECK(sl->expert_slot.count(0));
            CHECK(sl->expert_slot.count(1));
        }
    }
}

static void test_io_boundary() {
    fixture f; f.fill();
    CHECK(f.stream.resize_slots(6));
    // A stale final queue entry must wake a resize waiter even if no read completes afterward.
    {
        std::lock_guard<std::mutex> lock(f.stream.mtx);
        f.stream.q_demand.push_back({f.stream.layers[0].get(), 0, 999, 0, 0, 0});
        f.stream.start_workers_locked();
        f.stream.cv_work.notify_all();
    }
    CHECK(f.stream.resize_slots(7)); f.verify();

    // Exercise readers owning real destination pointers while resize drains demand work.
    FILE * file = tmpfile(); CHECK(file);
    {
        fixture io;
        const size_t slab = io.stream.max_nb_expert;
        std::vector<uint8_t> data(slab * 16, 0x42);
        CHECK(fwrite(data.data(), 1, data.size(), file) == data.size());
        CHECK(fflush(file) == 0);
        io.stream.files.emplace_back(new llama_file(file));
        {
            std::lock_guard<std::mutex> lock(io.stream.mtx);
            for (auto & sl : io.stream.layers) {
                io.stream.reserve_slot_locked(*sl, 0, 0);
                sl->slot_pending[0] = sl->weights.size();
                for (size_t wi = 0; wi < sl->weights.size(); ++wi) {
                    io.stream.q_demand.push_back({sl.get(), 0, 0, (int32_t) wi, sl->slot_gen[0], io.stream.allocation_epoch});
                }
            }
            io.stream.start_workers_locked();
            io.stream.cv_work.notify_all();
        }
        CHECK(io.stream.resize_slots(8));
        CHECK(io.stream.workers_active == 0 && io.stream.q_demand.empty());
        for (const auto & sl : io.stream.layers) {
            CHECK(sl->slot_state[0] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            for (const auto & wt : sl->weights) {
                std::vector<uint8_t> bytes(wt.nb_expert);
                ggml_backend_tensor_get(wt.cache, bytes.data(), 0, bytes.size());
                CHECK(bytes == std::vector<uint8_t>(wt.nb_expert, 0x42));
            }
        }
    }
    fclose(file);
}

static void test_whole_graph() {
    fixture f;
    auto & sl = *f.stream.layers[0];
    auto * weight = sl.weights[0].cache;
    std::vector<float> values(ggml_nelements(weight), 0.25f);
    ggml_backend_tensor_set(weight, values.data(), 0, ggml_nbytes(weight));
    sl.slot_expert[0] = 0; sl.slot_state[0] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
    sl.expert_slot[0] = 0;
    ggml_backend_ptr cpu{ggml_backend_cpu_init()};
    ggml_backend_cpu_set_n_threads(cpu.get(), 1);
    for (int size : {4, 9, 3, 7, 4}) {
        CHECK(f.stream.resize_slots(size));
        ggml_context_ptr ctx{ggml_init({ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true})};
        auto * inp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 1);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
        auto * out = ggml_mul_mat_id(ctx.get(), weight, inp, ids);
        auto * gf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_ptr buf{ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get())};
        CHECK(buf);
        std::vector<float> in(32, 1.0f); int id = 0;
        ggml_backend_tensor_set(inp, in.data(), 0, 32*sizeof(float));
        ggml_backend_tensor_set(ids, &id, 0, sizeof(id));
        CHECK(ggml_backend_graph_compute(cpu.get(), gf) == GGML_STATUS_SUCCESS);
        float result;
        ggml_backend_tensor_get(out, &result, 0, sizeof(result));
        CHECK(result == 8.0f);
    }
}

// In place: the first grow allocates, later transitions stay inside that allocation. CPU buffers
// resize logically (Metal also re-wraps views and releases the tail), so this covers the slot
// bookkeeping: no allocation, retained slabs byte-exact, tail residents moved into free head slots.
static void test_inplace() {
    setenv("LLAMA_MOE_STREAM_INPLACE", "1", 1);
    fixture f;
    unsetenv("LLAMA_MOE_STREAM_INPLACE");
    CHECK(f.stream.inplace);

    f.fill(); f.verify();
    CHECK(f.stream.resize_slots(12)); f.verify(); // beyond the allocation: the copying path
    CHECK(f.stream.n_inplace_layers == 0);
    f.cap_slots = 12;

    // make the tail the hottest, so a shrink to 5 keeps experts 7..11 and must move every one
    f.fill();
    for (auto & sl : f.stream.layers) {
        for (uint32_t e = 0; e < 12; ++e) {
            sl->route_hotness[e] = e + 1;
        }
    }
    f.probe.fail_at = 0; // any allocation now fails the resize
    CHECK(f.stream.resize_slots(5)); f.verify();
    CHECK(f.stream.n_inplace_layers == 3);
    for (const auto & sl : f.stream.layers) {
        for (uint32_t s = 0; s < 5; ++s) {
            CHECK(sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            CHECK(sl->slot_expert[s] >= 7);
        }
    }

    CHECK(f.stream.resize_slots(12)); f.verify(); // back up, still inside the allocation
    CHECK(f.stream.n_inplace_layers == 3);
    for (const auto & sl : f.stream.layers) {
        CHECK(sl->expert_slot.size() == 5);
        for (uint32_t s = 5; s < 12; ++s) {
            CHECK(sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY);
        }
    }
    CHECK(f.probe.fail_at == 0); // nothing was allocated

    f.probe.fail_at = -1;
    CHECK(f.stream.resize_slots(14)); // past the allocation: copying again
    CHECK(f.stream.n_inplace_layers == 0);
    f.cap_slots = 0;
    f.verify();
}

int main() {
    try {
            test_policy();
        test_residency_and_failures();
        test_io_boundary();
        test_whole_graph();
        test_inplace();
        puts("expert cache resize: PASS");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
