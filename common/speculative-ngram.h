#pragma once

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// A bounded, per-sequence index into VERIFIED token history. Like ngram-mod, it uses a
// direct-mapped hash table; unlike a token-only table, a hit is checked against the actual
// history before use. Collisions lose opportunities, never invent a match. The default index
// is 1 MiB and incremental updates avoid scanning a growing context on every decode step.
class common_speculative_ngram {
public:
    explicit common_speculative_ngram(size_t n_match, size_t table_size = 256*1024)
        : n_match(n_match), entries(table_size, empty) {
        if (n_match == 0 || table_size == 0) {
            throw std::invalid_argument("MTP n-gram lookup requires a positive match length and table size");
        }
    }

    void reset() {
        std::fill(entries.begin(), entries.end(), empty);
        indexed = 0;
        history_size = 0;
    }

    size_t size_bytes() const { return entries.size()*sizeof(entries[0]); }

    // prompt excludes the already sampled anchor. Neither the MTP prefix nor the generated
    // suffix is inserted into the index: rejected proposals must not train future lookups.
    std::vector<llama_token> draft(
            const std::vector<llama_token> & prompt, llama_token anchor,
            const std::vector<llama_token> & prefix, size_t n_max) {
        const size_t n_history = prompt.size() + 1;
        if (n_history > (size_t) std::numeric_limits<uint32_t>::max()) {
            return {};
        }
        if (n_history < history_size) {
            reset();
        }
        history_size = n_history;

        const auto token_at = [&](size_t i) { return i < prompt.size() ? prompt[i] : anchor; };
        const auto hash = [&](auto token, size_t first) {
            uint64_t h = 0;
            for (size_t i = 0; i < n_match; ++i) {
                h = h*6364136223846793005ULL + (uint32_t) token(first + i);
            }
            return h % entries.size();
        };

        // Each entry needs a known successor. Include the anchor, which was already sampled by
        // the target, but do not index the hypothetical continuation supplied in prefix.
        for (; indexed + n_match < n_history; ++indexed) {
            entries[hash(token_at, indexed)] = (uint32_t) indexed;
        }

        if (n_max == 0 || n_history + prefix.size() < n_match) {
            return {};
        }

        std::vector<llama_token> window;
        window.reserve(n_match + n_max);
        const size_t n_input = n_history + prefix.size();
        for (size_t i = n_input - n_match; i < n_input; ++i) {
            window.push_back(i < n_history ? token_at(i) : prefix[i - n_history]);
        }
        const auto window_at = [&](size_t i) { return window[i]; };

        std::vector<llama_token> result;
        result.reserve(n_max);
        for (size_t i = 0; i < n_max; ++i) {
            const uint32_t pos = entries[hash(window_at, i)];
            if (pos == empty || (size_t) pos + n_match >= n_history) {
                break;
            }
            bool match = true;
            for (size_t j = 0; j < n_match; ++j) {
                if (token_at((size_t) pos + j) != window[i + j]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                break;
            }
            const llama_token next = token_at((size_t) pos + n_match);
            result.push_back(next);
            window.push_back(next);
        }
        return result;
    }

private:
    static constexpr uint32_t empty = std::numeric_limits<uint32_t>::max();
    size_t n_match;
    std::vector<uint32_t> entries;
    size_t indexed = 0;
    size_t history_size = 0;
};
