// A small, synthetic CPU model exercises actual scheduler destruction/reservation with pending
// logits and populated KV. No downloaded model, Metal device or inference server is required.
#include "llama.h"
#include "llama-cpp.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-memory-phase.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

static void test_memory_types() {
    CHECK(!llama_memory_phase_buft_uses_system_ram(nullptr));
    CHECK(llama_memory_phase_buft_uses_system_ram(ggml_backend_cpu_buffer_type()));

    // Fake non-host buffers catch unsafe admission of discrete/unknown memory.
    // No backend or GPU buffer is initialized by these policy checks.
    ggml_backend_reg reg{};
    reg.iface.get_name = [](ggml_backend_reg_t r) { return (const char *) r->context; };
    ggml_backend_device dev{};
    dev.reg = &reg;
    ggml_backend_buffer_type buft{};
    buft.device = &dev;
    buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
    for (const char * name : {"CUDA", "Vulkan", "unknown", "MTL", "Metal"}) {
        reg.context = (void *) name;
        bool expected = false;
#if defined(__APPLE__) && defined(__aarch64__)
        expected = strcmp(name, "MTL") == 0 || strcmp(name, "Metal") == 0;
#endif
        CHECK(llama_memory_phase_buft_uses_system_ram(&buft) == expected);
    }
    buft.device = nullptr;
    CHECK(!llama_memory_phase_buft_uses_system_ram(&buft));
    buft.iface.is_host = [](ggml_backend_buffer_type_t) { return true; };
    CHECK(llama_memory_phase_buft_uses_system_ram(&buft));

    // Check real registration names as well, without creating a Metal context
    // or submitting GPU work. This catches the original MTL vs Metal failure.
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const auto actual = ggml_backend_dev_get(i);
        const auto actual_reg = ggml_backend_dev_backend_reg(actual);
        const char * name = ggml_backend_reg_name(actual_reg);
        const auto actual_buft = ggml_backend_dev_buffer_type(actual);
        if (strcmp(name, "CPU") == 0 || strcmp(name, "BLAS") == 0
#if defined(__APPLE__) && defined(__aarch64__)
                || strcmp(name, "MTL") == 0 || strcmp(name, "Metal") == 0
#endif
        ) {
            CHECK(llama_memory_phase_buft_uses_system_ram(actual_buft));
            printf("memory policy: %s / %s accepted\n", name, ggml_backend_buft_name(actual_buft));
        }
    }
}

static void init_tensor(ggml_tensor * tensor, void *) {
    // init_from_user also materializes optional RoPE factors; zero would divide by zero in RoPE.
    const bool norm = strstr(tensor->name, "norm") != nullptr || strstr(tensor->name, "rope") != nullptr;
    const size_t n = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> data(n);
        for (size_t i = 0; i < n; ++i) { data[i] = norm ? 1.0f : 0.03f * sinf((float) i); }
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        CHECK(tensor->type == GGML_TYPE_F16);
        std::vector<ggml_fp16_t> data(n);
        for (size_t i = 0; i < n; ++i) { data[i] = ggml_fp32_to_fp16(norm ? 1.0f : 0.03f * sinf((float) i)); }
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    }
}

static std::vector<uint8_t> state(llama_context * ctx) {
    std::vector<uint8_t> data(llama_state_get_size(ctx));
    CHECK(llama_state_get_data(ctx, data.data(), data.size()) == data.size());
    return data;
}

static std::vector<float> run(llama_context * ctx, int pos, int count) {
    auto batch = llama_batch_init(count, 0, 1);
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        batch.token[i] = 3 + (pos + i) % 64;
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    const int rc = llama_decode(ctx, batch);
    CHECK(rc == 0);
    const float * logits = llama_get_logits(ctx);
    std::vector<float> result(logits, logits + 128*count);
    llama_batch_free(batch);
    return result;
}

int main() {
    try {
        test_memory_types();
        // Model device list below selects CPU; the linked registry may enumerate other devices.
        gguf_context_ptr meta{gguf_init_empty()};
        gguf_set_val_str(meta.get(), "general.architecture", "llama");
        gguf_set_val_str(meta.get(), "tokenizer.ggml.model", "no_vocab");
        gguf_set_val_u32(meta.get(), "llama.vocab_size", 128);
        gguf_set_val_u32(meta.get(), "llama.context_length", 256);
        gguf_set_val_u32(meta.get(), "llama.embedding_length", 64);
        gguf_set_val_u32(meta.get(), "llama.block_count", 2);
        gguf_set_val_u32(meta.get(), "llama.feed_forward_length", 128);
        gguf_set_val_u32(meta.get(), "llama.attention.head_count", 1);
        gguf_set_val_u32(meta.get(), "llama.attention.head_count_kv", 1);
        gguf_set_val_f32(meta.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        llama_model_ptr model{llama_model_init_from_user(meta.get(), init_tensor, nullptr, mp)};
        CHECK(model);
        auto cp = llama_context_default_params();
        cp.n_ctx = 256; cp.n_batch = 128; cp.n_ubatch = 128;
        cp.n_threads = 1; cp.n_threads_batch = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context_ptr target{llama_init_from_model(model.get(), cp)};
        llama_context_ptr draft{llama_init_from_model(model.get(), cp)};
        CHECK(target && draft);
        int pos = 0;
        for (int cycle = 0; cycle < 3; ++cycle) {
            const auto logits = run(target.get(), pos, 16);
            const auto dlogits = run(draft.get(), pos, 16);
            if (logits != dlogits) {
                for (size_t i = 0; i < logits.size(); ++i) {
                    if (logits[i] != dlogits[i]) {
                        fprintf(stderr, "initial logits mismatch at %zu: %.9g / %.9g\n", i, logits[i], dlogits[i]);
                        break;
                    }
                }
            }
            CHECK(logits == dlogits);
            pos += 16;
            const auto before = state(target.get());
            const auto dbefore = state(draft.get());
            const uint64_t large = llama_compute_buffer_size(target.get()) + llama_compute_buffer_size(draft.get());
            CHECK(llama_memory_phase_transition(target.get(), draft.get(), 4, 4, 0));
            const uint64_t small = llama_compute_buffer_size(target.get()) + llama_compute_buffer_size(draft.get());
            CHECK(small < large);
            CHECK(llama_n_batch(target.get()) == 128 && llama_n_ubatch(target.get()) == 4);
            CHECK(state(target.get()) == before && state(draft.get()) == dbefore);
            CHECK(memcmp(llama_get_logits(target.get()), logits.data(), logits.size()*sizeof(float)) == 0);
            CHECK(memcmp(llama_get_logits(draft.get()), dlogits.data(), dlogits.size()*sizeof(float)) == 0);
            // Invalid capacities must fail before changing either context.
            CHECK(!llama_memory_phase_transition(target.get(), draft.get(), 129, 4, 0));
            CHECK(!llama_memory_phase_transition(target.get(), draft.get(), 4, 0, 0));
            CHECK(state(target.get()) == before && state(draft.get()) == dbefore);
            const auto small_logits = run(target.get(), pos, 4);
            CHECK(llama_memory_phase_transition(target.get(), draft.get(), 128, 128, 0));
            const auto large_logits = run(draft.get(), pos, 4);
            if (small_logits != large_logits) {
                double max_diff = 0;
                size_t nonfinite = 0;
                for (size_t i = 0; i < small_logits.size(); ++i) {
                    max_diff = std::max(max_diff, (double) fabsf(small_logits[i] - large_logits[i]));
                    nonfinite += !std::isfinite(small_logits[i]) || !std::isfinite(large_logits[i]);
                }
                fprintf(stderr, "phase logits difference: max_abs=%.9g nonfinite=%zu\n", max_diff, nonfinite);
            }
            CHECK(small_logits == large_logits);
            pos += 4;
            printf("workspace cycle %d: %llu -> %llu compute bytes; state/logits preserved\n", cycle,
                    (unsigned long long) large, (unsigned long long) small);
        }
        puts("memory phase: PASS");
        return 0;
    } catch (const std::exception & err) {
        fprintf(stderr, "%s\n", err.what());
        return 1;
    }
}
