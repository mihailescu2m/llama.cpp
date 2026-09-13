#pragma once

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <random>
#include <unordered_map>
#include <vector>

struct server_speculative_sample {
    llama_token token;
    const llama_token_data_array * candidates;
};

// Sampling and accepting are supplied by the server so this exact verifier can also be tested
// with small, known distributions. sample(i) returns the target's post-chain distribution at i.
template <typename Sample, typename Accept, typename IsEog>
std::vector<llama_token> server_speculative_rejection(
        const std::vector<llama_token> & draft,
        const std::vector<std::vector<llama_token_data>> & draft_dist,
        std::mt19937 & rng, Sample sample, Accept accept_token, IsEog is_eog) {
    assert(draft.size() == draft_dist.size());
    std::vector<llama_token> result;
    result.reserve(draft.size() + 1);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::unordered_map<llama_token, float> q;
    std::vector<float> resid;

    for (size_t i = 0; i < draft.size(); ++i) {
        const auto target = sample(i);
        const auto * cur_p = target.candidates;
        q.clear();
        for (const auto & c : draft_dist[i]) {
            q[c.id] = c.p;
        }

        float p_dft = 0.0f;
        float p_tgt = 0.0f;
        const auto it_dft = q.find(draft[i]);
        if (it_dft != q.end()) {
            p_dft = it_dft->second;
        }
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_tgt = cur_p->data[k].p;
                break;
            }
        }

        // EOG obeys the same min(1,p/q) rule; force-rejecting it biases the output distribution.
        const bool keep = p_dft > 0.0f && dist(rng)*p_dft < std::min(p_dft, p_tgt);
        if (keep) {
            accept_token(draft[i]);
            result.push_back(draft[i]);
            if (is_eog(draft[i])) {
                return result;
            }
            continue;
        }

        resid.assign(cur_p->size, 0.0f);
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const auto it = q.find(cur_p->data[k].id);
            const float r = cur_p->data[k].p - (it != q.end() ? it->second : 0.0f);
            if (r > 0.0f) {
                resid[k] = r;
                sum += r;
            }
        }

        llama_token token = target.token;
        if (sum > 0.0f) {
            float u = dist(rng)*sum;
            for (size_t k = 0; k < cur_p->size; ++k) {
                u -= resid[k];
                if (u <= 0.0f) {
                    token = cur_p->data[k].id;
                    break;
                }
            }
        }
        accept_token(token);
        result.push_back(token);
        return result;
    }

    const auto target = sample(draft.size());
    accept_token(target.token);
    result.push_back(target.token);
    return result;
}
