// Exercise disk indexing and actual target/draft restoration, including missing draft files.
#include "../tools/server/server-task.h"
#include "llama-cpp.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

struct temp_dir {
    std::filesystem::path path;
    temp_dir() {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int i = 0; i < 100; ++i) {
            auto candidate = std::filesystem::temp_directory_path() / ("llama-prompt-test-" + std::to_string(tick) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(candidate)) { path = candidate; return; }
        }
        throw std::runtime_error("cannot create temporary test directory");
    }
    ~temp_dir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

static void init_tensor(ggml_tensor * t, void *) {
    CHECK(t->type == GGML_TYPE_F32);
    const bool norm = strstr(t->name, "norm") || strstr(t->name, "rope");
    std::vector<float> data(ggml_nelements(t));
    for (size_t i = 0; i < data.size(); ++i) { data[i] = norm ? 1.0f : 0.03f*sinf((float) i); }
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

int main() {
    try {
        gguf_context_ptr meta{gguf_init_empty()};
        gguf_set_val_str(meta.get(), "general.architecture", "llama");
        gguf_set_val_str(meta.get(), "tokenizer.ggml.model", "no_vocab");
        gguf_set_val_u32(meta.get(), "llama.vocab_size", 128);
        gguf_set_val_u32(meta.get(), "llama.context_length", 32);
        gguf_set_val_u32(meta.get(), "llama.embedding_length", 64);
        gguf_set_val_u32(meta.get(), "llama.block_count", 1);
        gguf_set_val_u32(meta.get(), "llama.feed_forward_length", 128);
        gguf_set_val_u32(meta.get(), "llama.attention.head_count", 1);
        gguf_set_val_u32(meta.get(), "llama.attention.head_count_kv", 1);
        gguf_set_val_f32(meta.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        mp.devices = devices; mp.n_gpu_layers = 0;
        llama_model_ptr model{llama_model_init_from_user(meta.get(), init_tensor, nullptr, mp)};
        CHECK(model);
        auto cp = llama_context_default_params();
        cp.n_ctx = 32; cp.n_batch = 32; cp.n_ubatch = 32;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context_ptr target{llama_init_from_model(model.get(), cp)};
        llama_context_ptr draft{llama_init_from_model(model.get(), cp)};
        CHECK(target && draft);
        llama_tokens tokens{3, 4, 5, 6};
        auto batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
        CHECK(llama_decode(target.get(), batch) == 0 && llama_decode(draft.get(), batch) == 0);
        temp_dir dir;
        for (int scenario = 0; scenario < 4; ++scenario) {
            const auto path = dir.path / std::to_string(scenario);
            std::filesystem::create_directory(path);
            server_prompt_cache saved(0, 32, path.string());
            server_prompt source;
            source.tokens = server_tokens(tokens, false);
            auto & ckpt = source.checkpoints.emplace_back();
            ckpt.n_tokens = 3; ckpt.pos_min = 0; ckpt.pos_max = 2;
            ckpt.data_tgt = {9, 8, 7}; ckpt.data_dft = {6, 5}; ckpt.data_spec = {4};
            auto * allocated = saved.alloc(source, 1024, 1024);
            CHECK(allocated && allocated->prompt.checkpoints.empty());
            auto & st = *allocated;
            st.data.hbnd = {1, 2, 3, 4}; // an independent boundary survives missing KV
            CHECK(saved.save_direct(st, target.get(), scenario == 3 ? nullptr : draft.get(), 0, source.checkpoints));
            CHECK(st.prompt.checkpoints.empty() && source.checkpoints.size() == 1);
            if (scenario == 1) { CHECK(std::filesystem::remove(st.path_drft())); }
            if (scenario == 2) { std::filesystem::resize_file(st.path_drft(), 4); }
            server_prompt_cache restored(0, 32, path.string());
            restored.restore_index(target.get(), false);
            CHECK(restored.states.size() == 1);
            CHECK(restored.states.front().prompt.checkpoints.empty());
            server_prompt prompt;
            server_tokens next(llama_tokens{3, 4, 5, 6, 7}, false);
            bool got_target = false, got_draft = true;
            std::vector<uint8_t> boundary;
            CHECK(restored.load(prompt, next, target.get(), draft.get(), 0, &boundary, &got_target, &got_draft));
            CHECK(got_target && got_draft == (scenario == 0));
            CHECK(boundary == st.data.hbnd && prompt.tokens.size() == tokens.size());
            CHECK(prompt.checkpoints.size() == 1);
            CHECK(prompt.checkpoints.front().data_tgt == ckpt.data_tgt);
            CHECK(prompt.checkpoints.front().data_dft == ckpt.data_dft);
            CHECK(prompt.checkpoints.front().data_spec == ckpt.data_spec);
            CHECK(restored.states.front().prompt.checkpoints.empty());
            got_target = got_draft = true;
            CHECK(restored.load(prompt, next, target.get(), draft.get(), 0, &boundary, &got_target, &got_draft));
            CHECK(!got_target && !got_draft); // cache miss must not retain stale success flags
        }
        puts("prompt cache draft restoration: PASS");
        return 0;
    } catch (const std::exception & err) { fprintf(stderr, "%s\n", err.what()); return 1; }
}
