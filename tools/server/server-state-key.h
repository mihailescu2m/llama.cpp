#pragma once

#include "common.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

// Conservative LOCAL file identity, not a content digest: changing/replacing a file invalidates
// reuse, even if its length and GGUF header stay identical. Nanosecond change times catch in-place
// payload writes with a restored mtime. This avoids reading hundreds of GiB just to open a cache.
// Moving/copying/touching weights can cause harmless cache misses; live weight mutation is unsupported.
using server_state_file_stamp = std::array<uint64_t, 8>;

inline bool server_state_file_identity(const std::filesystem::path & path, server_state_file_stamp & out) {
#if defined(_WIN32)
    HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) { return false; }
    BY_HANDLE_FILE_INFORMATION info = {};
    FILE_BASIC_INFO basic = {};
    const bool ok = GetFileInformationByHandle(file, &info) &&
                    GetFileInformationByHandleEx(file, FileBasicInfo, &basic, sizeof(basic));
    CloseHandle(file);
    if (!ok || (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) { return false; }
    out = { info.dwVolumeSerialNumber,
            ((uint64_t) info.nFileIndexHigh << 32) | info.nFileIndexLow,
            ((uint64_t) info.nFileSizeHigh << 32) | info.nFileSizeLow,
            (uint64_t) basic.LastWriteTime.QuadPart, 0, (uint64_t) basic.ChangeTime.QuadPart, 0,
            (uint64_t) basic.CreationTime.QuadPart };
#else
    struct stat info = {};
    if (stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0) { return false; }
#if defined(__APPLE__)
    const auto mtime = info.st_mtimespec;
    const auto ctime = info.st_ctimespec;
#else
    const auto mtime = info.st_mtim;
    const auto ctime = info.st_ctim;
#endif
    out = { (uint64_t) info.st_dev, (uint64_t) info.st_ino, (uint64_t) info.st_size,
            (uint64_t) mtime.tv_sec, (uint64_t) mtime.tv_nsec,
            (uint64_t) ctime.tv_sec, (uint64_t) ctime.tv_nsec, 0 };
#endif
    return true;
}

// A state is replayable only for the same files, graph configuration and cache layout.
// Keep this CPU-only so identity/invalidation regressions can be tested without loading a model.
inline std::string server_model_state_id(const std::string & path_first_shard, const common_params & p) {
    // Per-request adapter scales are not part of a model-global key. Until the disk formats carry
    // that identity per entry, refuse automatic disk reuse for LoRA rather than guess.
    if (!p.lora_adapters.empty()) {
        return {};
    }

    uint64_t h = 1469598103934665603ull; // FNV-1a
    auto mix = [&h](const void * data, size_t n) {
        const uint8_t * p = (const uint8_t *) data;
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    };

    auto mix_string = [&mix](const std::string & value) {
        const uint64_t size = value.size();
        mix(&size, sizeof(size));
        mix(value.data(), value.size());
    };
    auto mix_file = [&mix, &mix_string](const std::string & path) {
        std::error_code ec;
        const auto canonical = std::filesystem::canonical(path, ec);
        if (ec) { return false; }
        server_state_file_stamp before, after;
        if (!server_state_file_identity(canonical, before)) { return false; }
        std::ifstream in(canonical, std::ios::binary);
        if (!in) { return false; }
        const size_t n = (size_t) std::min<uint64_t>(before[2], 1u << 20);
        std::vector<char> head(n);
        if (n && !in.read(head.data(), n)) { return false; }
        if (!server_state_file_identity(canonical, after) || before != after) { return false; }
        mix_string(canonical.generic_string());
        mix(before.data(), before.size() * sizeof(before[0]));
        mix(head.data(), head.size());
        return true;
    };

    // llama_split_prefix/llama_split_path own this parsing: they validate the suffix and build
    // it back with the same format string, so the two can never drift. Hand-rolled arithmetic
    // here previously dropped one character and hashed names that open nothing.
    auto mix_weights = [&mix_file](const std::string & path_first) -> bool {
        std::vector<std::string> paths = { path_first };
        {
            const std::string & f = path_first;
            const size_t pos = f.rfind("-of-");
            if (pos != std::string::npos && f.size() >= pos + 9) {
                const int total = atoi(f.substr(pos + 4, 5).c_str());
                std::vector<char> prefix(f.size() + 1);
                if (total > 1 && total < 1000 &&
                    llama_split_prefix(prefix.data(), prefix.size(), f.c_str(), 0, total) > 0) {
                    paths.clear();
                    for (int i = 0; i < total; ++i) {
                        std::vector<char> buf(f.size() + 32);
                        if (llama_split_path(buf.data(), buf.size(), prefix.data(), i, total) <= 0) {
                            return false;
                        }
                        paths.push_back(buf.data());
                    }
                }
            }
        }

        for (const auto & f : paths) {
            if (!mix_file(f)) { return false; }
        }

        return true;
    };

    if (!mix_weights(path_first_shard)) {
        return {};
    }

    const int32_t cfg_i[] = {
        p.n_ctx, p.n_parallel, (int32_t) p.cache_type_k, (int32_t) p.cache_type_v,
        (int32_t) p.flash_attn_type, (int32_t) p.rope_scaling_type, p.yarn_orig_ctx,
        p.grp_attn_n, p.grp_attn_w, (int32_t) p.swa_full, (int32_t) p.kv_unified,
        p.kv_unified_per_slot, p.n_batch, p.n_ubatch, (int32_t) p.attention_type,
        (int32_t) p.pooling_type, (int32_t) p.no_kv_offload, (int32_t) p.no_op_offload,
    };
    const float cfg_f[] = {
        p.rope_freq_base, p.rope_freq_scale, p.yarn_ext_factor, p.yarn_attn_factor,
        p.yarn_beta_fast, p.yarn_beta_slow,
    };
    mix(cfg_i, sizeof(cfg_i));
    mix(cfg_f, sizeof(cfg_f));

    // Overrides affect graph construction and control vectors alter cached hidden states.
    const uint64_t n_overrides = std::count_if(p.kv_overrides.begin(), p.kv_overrides.end(),
            [](const llama_model_kv_override & kv) { return kv.key[0] != '\0'; });
    mix(&n_overrides, sizeof(n_overrides));
    for (const auto & kv : p.kv_overrides) {
        if (kv.key[0] == '\0') { break; }
        mix_string(std::string(kv.key, std::find(kv.key, kv.key + sizeof(kv.key), '\0')));
        const int32_t tag = kv.tag;
        mix(&tag, sizeof(tag));
        switch (kv.tag) {
            case LLAMA_KV_OVERRIDE_TYPE_INT:   mix(&kv.val_i64, sizeof(kv.val_i64));   break;
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT: mix(&kv.val_f64, sizeof(kv.val_f64));   break;
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:  mix(&kv.val_bool, sizeof(kv.val_bool)); break;
            case LLAMA_KV_OVERRIDE_TYPE_STR:
                mix_string(std::string(kv.val_str, std::find(kv.val_str, kv.val_str + sizeof(kv.val_str), '\0')));
                break;
            default: return {};
        }
    }
    const uint64_t n_control = p.control_vectors.size();
    mix(&n_control, sizeof(n_control));
    for (const auto & cv : p.control_vectors) {
        if (!mix_file(cv.fname)) { return {}; }
        mix(&cv.strength, sizeof(cv.strength));
    }
    mix(&p.control_vector_layer_start, sizeof(p.control_vector_layer_start));
    mix(&p.control_vector_layer_end, sizeof(p.control_vector_layer_end));

    // These graph switches change values or layouts without changing common_params. Keep the
    // explicit list alongside the semantic epoch, and extend it when adding such a switch.
    for (const char * name : {
            "LLAMA_QSA_GATHER", "LLAMA_QSA_GATHER_MIN_KV", "LLAMA_QSA_NO_POOLED_CACHE",
            "LLAMA_DSV4_UNION", "LLAMA_DSV4_UNION_MIN_NCSA", "LLAMA_SPEC_DRAFT_UBATCH" }) {
        mix_string(name);
        const char * value = std::getenv(name);
        const uint8_t present = value != nullptr;
        mix(&present, sizeof(present));
        mix_string(value ? value : "");
    }

    // The drafter writes its own KV and boundary carryover into sibling files keyed by this same
    // id, and none of it is covered by the target hashes above. A different draft model, a
    // different drafting method, or a different draft cache type all make those bytes mean
    // something else, so they belong in the key.
    // The type list is variable length, so mix its count first: without it two different lists
    // that spell the same bytes collide, which is the one thing this id may not do.
    const auto & sp = p.speculative;
    const int32_t n_types = (int32_t) sp.types.size();
    mix(&n_types, sizeof(n_types));
    for (const auto t : sp.types) {
        const int32_t v = (int32_t) t;
        mix(&v, sizeof(v));
    }
    // Hash the draft weights the same way as the target's, not just the path: a requant in place
    // keeps the name and changes every byte the restored draft KV was built from. An unreadable
    // draft is only fatal once one is configured.
    if (!sp.draft.mparams.path.empty() && !mix_weights(sp.draft.mparams.path)) {
        return {};
    }
    const int32_t dft_i[] = {
        (int32_t) sp.draft.cache_type_k, (int32_t) sp.draft.cache_type_v, sp.draft.n_max,
        sp.mtp_ngram_n_max, sp.mtp_ngram_n_max > 0 ? sp.ngram_mod.n_match : 0,
    };
    mix(dft_i, sizeof(dft_i));

    // Bump when a change alters what the stored bytes mean without changing any field above -
    // a new cache layout, a graph change that redefines a cached value. Restoring an old state
    // across such a change is the silent-nonsense case and nothing else here would catch it.
    const int32_t state_semantics_version = 4; // safe selection for partial/multi-sequence QSA blocks
    mix(&state_semantics_version, sizeof(state_semantics_version));

    char out[17];
    snprintf(out, sizeof(out), "%016llx", (unsigned long long) h);
    return std::string(out);
}
