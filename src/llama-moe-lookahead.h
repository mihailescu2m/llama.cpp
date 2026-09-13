#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

// Select at most K unique experts from the last row's next-router logits, ranked by
// sqrt(softplus(x)) + bias. This chooses IO hints only, never actual routed experts.
inline void llama_moe_lookahead_select(
        const float * logits, uint32_t n_expert, uint32_t n_rows, uint32_t k,
        const std::vector<float> & bias, std::vector<float> & scratch, std::vector<int32_t> & selected) {
    selected.clear();
    assert(bias.empty() || bias.size() >= n_expert);
    if (!n_expert || !n_rows || !k) {
        return;
    }
    const float * row = logits + (size_t) (n_rows - 1)*n_expert;
    scratch.resize(n_expert);
    for (uint32_t e = 0; e < n_expert; ++e) {
        const float x = row[e];
        const float score = std::sqrt(x > 20.0f ? x : log1pf(expf(x))) + (bias.empty() ? 0.0f : bias[e]);
        scratch[e] = std::isfinite(score) ? score : -INFINITY;
    }
    k = std::min(k, n_expert);
    selected.reserve(k);
    while (selected.size() < k) {
        int32_t best = -1;
        float score = -INFINITY;
        for (uint32_t e = 0; e < n_expert; ++e) {
            if (scratch[e] > score) {
                score = scratch[e];
                best = (int32_t) e;
            }
        }
        if (best < 0) {
            break;
        }
        selected.push_back(best);
        scratch[best] = -INFINITY;
    }
}
