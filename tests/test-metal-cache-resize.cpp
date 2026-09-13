// Small Metal allocations only: no model weights, server, or inference workload.
#include "../src/llama-moe-stream.h"
#include "ggml-metal.h"
#include <cstdio>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

static std::mutex hook_mutex;
static int fail_at = -1;
static bool throw_failure = false;
static int installed = 0;
static bool install_views(ggml_backend_buffer_t buf, const size_t * off, const size_t * size, int n) {
    std::lock_guard<std::mutex> lock(hook_mutex);
    if (fail_at >= 0 && fail_at-- == 0) {
        if (throw_failure) { throw std::bad_alloc(); }
        return false;
    }
    const bool ok = ggml_backend_metal_buffer_set_views(buf, off, size, n);
    if (ok) { ++installed; }
    return ok;
}

int main() {
    try {
        auto * dev = ggml_backend_dev_by_name("Metal");
        if (!dev) { dev = ggml_backend_dev_by_name("MTL0"); }
        if (!dev) { puts("Metal unavailable: SKIP"); return 77; }
        llama_moe_stream stream(3, 4, 1, false);
        stream.inplace = true;
        stream.warm_decode = false;
        ggml_context_ptr meta{ggml_init({ggml_tensor_overhead()*4, nullptr, true})};
        for (int il = 0; il < 3; ++il) {
            auto * wt = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, 32, 256, 16);
            stream.create_cache_tensor(il, ggml_backend_dev_buffer_type(dev), wt, 0, 0);
        }
        stream.alloc_bufs(false);
        for (auto & sl : stream.layers) {
            auto * t = sl->weights[0].cache;
            std::vector<float> data(ggml_nelements(t), (float) sl->il + 1);
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
            sl->slot_expert[0] = 0; sl->slot_state[0] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
            sl->expert_slot[0] = 0;
        }
        CHECK(stream.resize_slots(12)); // split allocation, shrunk/grown in place below
        CHECK(stream.resize_slots(4));
        auto verify = [&](uint32_t slots) {
            CHECK(stream.n_slots == slots && !stream.resizing);
            for (const auto & sl : stream.layers) {
                CHECK(sl->n_slots == slots && sl->weights[0].cache->ne[2] == slots);
                const int s = sl->expert_slot.at(0);
                std::vector<float> data(32*256);
                ggml_backend_tensor_get(sl->weights[0].cache, data.data(), s*data.size()*sizeof(float), data.size()*sizeof(float));
                for (float v : data) { CHECK(v == sl->il + 1); }
            }
            CHECK(stream.pending_views.empty() && stream.pending_release.empty());
        };
        stream.resize_set_views = install_views;
        for (bool throws : {false, true}) {
            fail_at = 1; throw_failure = throws; installed = 0;
            CHECK(!stream.resize_slots(12));
            CHECK(installed >= 1); // failure after another group has installed its new views
            verify(4);
            fail_at = -1;
            CHECK(stream.resize_slots(12)); verify(12);
            CHECK(stream.resize_slots(4)); verify(4);
        }
        CHECK(stream.resize_slots(12));
        fail_at = 0; throw_failure = false;
        CHECK(stream.resize_slots(4)); // a failed shrink keeps its safe superset views
        verify(4);
        puts("Metal resize rollback: PASS");
        return 0;
    } catch (const std::exception & err) { fprintf(stderr, "%s\n", err.what()); return 1; }
}
