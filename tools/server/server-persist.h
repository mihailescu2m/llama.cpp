#pragma once

// On-disk format shared by --slot-persist and the SSD context cache: a sequence state file (tokens
// and KV, llama_state_seq_save_file), a checkpoint sidecar (.ckpt), the drafter boundary (.hbnd),
// the draft KV (.drft) and a commit record (.gen) that is written last.

#include "server-task.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <list>
#include <string>
#include <vector>

// Context checkpoints are server state and are not part of llama_state_seq_save_file(). Keep
// them in a best-effort sidecar so an SWA slot can resume near the end of its prompt after a
// restart instead of reprocessing from position zero.
static constexpr uint32_t SLOT_CKPT_MAGIC     = 0x4b435453; // "STCK"
static constexpr uint32_t SLOT_CKPT_VERSION   = 2;
static constexpr uint32_t SLOT_CKPT_MAX       = 64;
static constexpr uint64_t SLOT_CKPT_MAX_BLOB  = 4ull  << 30;
static constexpr uint64_t SLOT_CKPT_MAX_TOTAL = 16ull << 30;

inline std::string slot_ckpt_path(const std::string & filepath) {
    return filepath + ".ckpt";
}

inline std::string slot_gen_path(const std::string & filepath) {
    return filepath + ".gen";
}

// Commit record tying a persisted state to its sidecars. The .hbnd and .drft files are written
// after state.bin is already renamed into place, so a crash in between leaves a NEW state beside
// OLD sidecars - and nothing in their own contents can tell: a boundary saved earlier in the same
// conversation can carry exactly the position the new prefix continues at, so the position check
// passes on a blob describing different KV. This file is removed before a save begins and written
// after it ends, so its presence is the statement that the whole set came from one save.
//
// It covers a crashed or killed process, which is the case that happens here. A machine crash can
// still reorder these writes in the page cache; fsync on every save would close that and costs far
// more than the resynchronized drafter it would save.
static constexpr uint32_t SLOT_GEN_MAGIC   = 0x4e454753; // "SGEN"
static constexpr uint32_t SLOT_GEN_VERSION = 1;

struct slot_gen_hdr {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tokens;   // the prompt the sidecars were captured against
    uint64_t sz_hbnd;    // 0 when that sidecar was not written
    uint64_t sz_drft;
};

inline void slot_gen_save(const std::string & filepath, uint64_t n_tokens, uint64_t sz_hbnd, uint64_t sz_drft) {
    const slot_gen_hdr hdr = { SLOT_GEN_MAGIC, SLOT_GEN_VERSION, n_tokens, sz_hbnd, sz_drft };

    const std::string path = slot_gen_path(filepath);
    std::ofstream out(path, std::ios::binary);
    if (!out || !out.write((const char *) &hdr, sizeof(hdr))) {
        // without it the next restore ignores the sidecars and resynchronizes - slower, not wrong
        std::remove(path.c_str());
    }
}

// Reads the record back. Returns false when the set cannot be trusted as one generation, which is
// the ordinary case on a first run and after any interrupted save.
inline bool slot_gen_load(const std::string & filepath, uint64_t n_tokens, slot_gen_hdr & hdr) {
    std::ifstream in(slot_gen_path(filepath), std::ios::binary);
    if (!in || !in.read((char *) &hdr, sizeof(hdr))) {
        return false;
    }

    return hdr.magic == SLOT_GEN_MAGIC && hdr.version == SLOT_GEN_VERSION && hdr.n_tokens == n_tokens;
}

inline uint64_t slot_ckpt_hash(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

template <typename T>
inline bool slot_ckpt_write(std::ofstream & file, const T & value) {
    file.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return (bool) file;
}

inline bool slot_ckpt_write_blob(std::ofstream & file, const std::vector<uint8_t> & blob) {
    const uint64_t size = blob.size();
    if (size > SLOT_CKPT_MAX_BLOB || !slot_ckpt_write(file, size)) {
        return false;
    }
    if (size > 0) {
        file.write(reinterpret_cast<const char *>(blob.data()), size);
    }
    return (bool) file;
}

template <typename T>
inline bool slot_ckpt_read(std::ifstream & file, T & value) {
    return (bool) file.read(reinterpret_cast<char *>(&value), sizeof(value));
}

// `left` is how many bytes remain in the file. Without it a truncated sidecar - power loss, a full
// disk, a killed server - can declare a 4 GiB blob and resize() commits that allocation BEFORE the
// read discovers the bytes are not there. The throw is caught, so it is not a crash, but a 4 GiB
// transient on a box already near its memory ceiling is enough to start paging, and a truncated
// sidecar is exactly what the situations that produce one leave behind.
inline bool slot_ckpt_read_blob(std::ifstream & file, std::vector<uint8_t> & blob, uint64_t left) {
    uint64_t size = 0;
    if (!slot_ckpt_read(file, size) || size > SLOT_CKPT_MAX_BLOB || size > left) {
        return false;
    }
    blob.resize(size);
    return size == 0 || (bool) file.read(reinterpret_cast<char *>(blob.data()), size);
}

// Returns bytes written. A failure is non-fatal because the main state file is already complete;
// removing the previous sidecar first ensures a failed or checkpoint-free save cannot reuse stale
// checkpoint data on the next restore.

inline size_t slot_ckpt_save(
        const std::string & filepath,
        const void * prompt_data,
        size_t prompt_size,
        size_t prompt_tokens,
        const std::list<common_prompt_checkpoint> & checkpoints) {
    const std::string path = slot_ckpt_path(filepath);
    const std::string path_tmp = path + ".tmp";

    std::error_code error;
    std::filesystem::remove(path_tmp, error);
    error.clear();
    std::filesystem::remove(path, error);

    if (checkpoints.empty()) {
        return 0;
    }

    const uint32_t count = (uint32_t) std::min<size_t>(checkpoints.size(), SLOT_CKPT_MAX);
    size_t skip = checkpoints.size() - count;
    uint64_t total = 0;
    for (const auto & checkpoint : checkpoints) {
        if (skip > 0) {
            --skip;
            continue;
        }
        if (checkpoint.data_tgt.size() > SLOT_CKPT_MAX_BLOB ||
            checkpoint.data_dft.size() > SLOT_CKPT_MAX_BLOB ||
            checkpoint.data_spec.size() > SLOT_CKPT_MAX_BLOB ||
            checkpoint.size() > SLOT_CKPT_MAX_TOTAL - total) {
            return 0;
        }
        total += checkpoint.size();
    }
    if (total > SLOT_CKPT_MAX_TOTAL) {
        return 0;
    }

    std::ofstream file(path_tmp, std::ios::binary | std::ios::trunc);
    if (!file) {
        return 0;
    }
    if (
        !slot_ckpt_write(file, SLOT_CKPT_MAGIC) ||
        !slot_ckpt_write(file, SLOT_CKPT_VERSION) ||
        !slot_ckpt_write(file, count) ||
        !slot_ckpt_write(file, (uint64_t) prompt_size) ||
        !slot_ckpt_write(file, (uint64_t) prompt_tokens) ||
        !slot_ckpt_write(file, slot_ckpt_hash(prompt_data, prompt_size))) {
        file.close();
        std::filesystem::remove(path_tmp, error);
        return 0;
    }

    skip = checkpoints.size() - count;
    for (const auto & checkpoint : checkpoints) {
        if (skip > 0) {
            --skip;
            continue;
        }

        if (!slot_ckpt_write(file, checkpoint.n_tokens) ||
            !slot_ckpt_write(file, checkpoint.pos_min) ||
            !slot_ckpt_write(file, checkpoint.pos_max) ||
            !slot_ckpt_write_blob(file, checkpoint.data_tgt) ||
            !slot_ckpt_write_blob(file, checkpoint.data_dft) ||
            !slot_ckpt_write_blob(file, checkpoint.data_spec)) {
            file.close();
            std::filesystem::remove(path_tmp, error);
            return 0;
        }
    }

    const std::streampos end = file.tellp();
    file.close();
    if (!file || end < 0) {
        std::filesystem::remove(path_tmp, error);
        return 0;
    }

    error.clear();
    std::filesystem::rename(path_tmp, path, error);
    if (error) {
        std::filesystem::remove(path_tmp, error);
        return 0;
    }

    return (size_t) end;
}

// Returns the number of recovered checkpoints. Missing, stale, or malformed sidecars preserve the
// upstream behavior: the checkpoint list stays empty and prompt reuse may fall back to a re-prefill.
inline size_t slot_ckpt_load(
        const std::string & filepath,
        const void * prompt_data,
        size_t prompt_size,
        size_t prompt_tokens,
        std::list<common_prompt_checkpoint> & checkpoints) {
    checkpoints.clear();

    const std::string path = slot_ckpt_path(filepath);
    std::error_code error;
    const uint64_t file_size = std::filesystem::file_size(path, error);
    if (error || file_size > SLOT_CKPT_MAX_TOTAL) {
        return 0;
    }

    try {
        std::ifstream file(path, std::ios::binary);
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t count = 0;
        uint64_t saved_prompt_size = 0;
        uint64_t saved_prompt_tokens = 0;
        uint64_t saved_prompt_hash = 0;
        if (!file ||
            !slot_ckpt_read(file, magic) || magic != SLOT_CKPT_MAGIC ||
            !slot_ckpt_read(file, version) || version != SLOT_CKPT_VERSION ||
            !slot_ckpt_read(file, count) || count > SLOT_CKPT_MAX ||
            !slot_ckpt_read(file, saved_prompt_size) || saved_prompt_size != prompt_size ||
            !slot_ckpt_read(file, saved_prompt_tokens) || saved_prompt_tokens != prompt_tokens ||
            !slot_ckpt_read(file, saved_prompt_hash) || saved_prompt_hash != slot_ckpt_hash(prompt_data, prompt_size)) {
            return 0;
        }

        std::list<common_prompt_checkpoint> loaded;
        for (uint32_t i = 0; i < count; ++i) {
            // recomputed per blob: each read advances the position, so the bound tightens
            auto left = [&]() -> uint64_t {
                const std::streampos at = file.tellg();
                return at < 0 || (uint64_t) at > file_size ? 0 : file_size - (uint64_t) at;
            };

            common_prompt_checkpoint checkpoint;
            if (!slot_ckpt_read(file, checkpoint.n_tokens) ||
                !slot_ckpt_read(file, checkpoint.pos_min) ||
                !slot_ckpt_read(file, checkpoint.pos_max) ||
                checkpoint.n_tokens < 0 || checkpoint.n_tokens > (int64_t) prompt_tokens ||
                checkpoint.pos_min < 0 || checkpoint.pos_max < checkpoint.pos_min ||
                !slot_ckpt_read_blob(file, checkpoint.data_tgt, left()) || checkpoint.data_tgt.empty() ||
                !slot_ckpt_read_blob(file, checkpoint.data_dft, left()) ||
                !slot_ckpt_read_blob(file, checkpoint.data_spec, left())) {
                return 0;
            }
            loaded.push_back(std::move(checkpoint));
        }

        if (file.peek() != std::ifstream::traits_type::eof()) {
            return 0;
        }

        checkpoints = std::move(loaded);
        return checkpoints.size();
    } catch (const std::exception &) {
        checkpoints.clear();
        return 0;
    }
}
