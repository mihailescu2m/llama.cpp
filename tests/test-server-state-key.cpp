#include "../tools/server/server-state-key.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstring>

namespace fs = std::filesystem;

struct fixture_dir {
    fs::path path;
    fixture_dir() {
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned i = 0; i < 100; ++i) {
            auto candidate = fs::temp_directory_path() /
                    ("llama-state-key-" + std::to_string(tick) + "-" + std::to_string(i));
            if (fs::create_directory(candidate)) { path = candidate; return; }
        }
        throw std::runtime_error("cannot create test fixture directory");
    }
    ~fixture_dir() {
        std::error_code ec;
        if (!path.empty()) { fs::remove_all(path, ec); }
    }
};

static void write_fixture(const fs::path & path) {
    std::ofstream out(path, std::ios::binary);
    std::vector<char> bytes((1u << 20) + 64, 'a');
    assert(out.write(bytes.data(), bytes.size()));
    out.close();
    assert(out);
}

static void edit_payload(const fs::path & path) {
    const auto mtime = fs::last_write_time(path);
    std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    out.seekp((1u << 20) + 5);
    out.put('b');
    out.close();
    assert(out);
    // Preserve length AND mtime. The change time must still invalidate the cached state.
    fs::last_write_time(path, mtime);
}

struct scoped_env {
    std::string name;
    std::string previous;
    bool existed;
    scoped_env(const char * key, const char * value) : name(key), existed(std::getenv(key) != nullptr) {
        if (existed) { previous = std::getenv(key); }
        set(value);
    }
    void set(const char * value) const {
#if defined(_WIN32)
        _putenv_s(name.c_str(), value ? value : "");
#else
        if (value) { setenv(name.c_str(), value, 1); } else { unsetenv(name.c_str()); }
#endif
    }
    ~scoped_env() { set(existed ? previous.c_str() : nullptr); }
};

int main() {
    fixture_dir dir;
    const auto target = dir.path / "target.gguf";
    const auto draft = dir.path / "draft.gguf";
    write_fixture(target);
    write_fixture(draft);
    common_params p;
    p.model.path = target.string();
    auto key = [&] { return server_model_state_id(p.model.path, p); };
    const auto base = key();
    assert(!base.empty() && key() == base);
    edit_payload(target);
    const auto edited = key();
    assert(edited != base);

    // Replacing a same-size, same-prefix artifact also invalidates, even with the old mtime.
    const auto replacement = dir.path / "replacement.gguf";
    write_fixture(replacement);
    fs::last_write_time(replacement, fs::last_write_time(target));
    fs::remove(target);
    fs::rename(replacement, target);
    assert(key() != edited);

    p.speculative.draft.mparams.path = draft.string();
    const auto with_draft = key();
    edit_payload(draft);
    assert(key() != with_draft);
    fs::remove(draft);
    assert(key().empty());
    p.speculative.draft.mparams.path.clear();

    // Every split is required, and a payload edit in a later shard must change the key.
    const auto first = dir.path / "split-00001-of-00002.gguf";
    const auto second = dir.path / "split-00002-of-00002.gguf";
    write_fixture(first);
    p.model.path = first.string();
    assert(key().empty());
    write_fixture(second);
    const auto split_key = key();
    assert(!split_key.empty());
    edit_payload(second);
    assert(key() != split_key);
    p.model.path = target.string();

    auto changes_key = [&](auto change) {
        common_params changed = p;
        change(changed);
        assert(server_model_state_id(changed.model.path, changed) != key());
    };
    changes_key([](common_params & q) { q.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; });
    changes_key([](common_params & q) { q.yarn_orig_ctx = 32768; });
    changes_key([](common_params & q) { q.yarn_ext_factor = 0.5f; });
    changes_key([](common_params & q) { q.yarn_attn_factor = 0.5f; });
    changes_key([](common_params & q) { q.yarn_beta_fast = 16; });
    changes_key([](common_params & q) { q.yarn_beta_slow = 2; });
    changes_key([](common_params & q) { q.speculative.draft.n_max++; });
    changes_key([](common_params & q) { q.speculative.mtp_ngram_n_max = 3; });
    changes_key([](common_params & q) { q.speculative.mtp_ngram_n_max = 3; q.speculative.ngram_mod.n_match = 8; });
    changes_key([](common_params & q) { q.speculative.draft.cache_type_k = GGML_TYPE_Q8_0; });
    changes_key([](common_params & q) { q.kv_unified = !q.kv_unified; });
    changes_key([](common_params & q) {
        llama_model_kv_override kv = {};
        std::strcpy(kv.key, "qwen4exp.rope.freq_base");
        kv.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kv.val_f64 = 10000;
        q.kv_overrides.push_back(kv);
    });
    for (const char * name : {
            "LLAMA_QSA_GATHER", "LLAMA_QSA_GATHER_MIN_KV", "LLAMA_QSA_NO_POOLED_CACHE",
            "LLAMA_DSV4_UNION", "LLAMA_DSV4_UNION_MIN_NCSA", "LLAMA_SPEC_DRAFT_UBATCH" }) {
        scoped_env env(name, "0");
        const auto disabled = key();
        env.set("1");
        assert(key() != disabled);
    }
    p.lora_adapters.push_back({});
    assert(key().empty());
    puts("server state key: PASS (payload edits, replacements, shards, draft, config, LoRA guard)");
}
