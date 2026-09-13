#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common.h"
#include "speculative.h"
#include "../tools/server/server-speculative.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

static void test_rejection() {
    const llama_tokens draft{10,11,12,13,14,15};
    std::vector<std::vector<llama_token_data>> q;
    for (auto t : draft) { q.push_back({{t, 0.0f, 1.0f}}); }

    // Force rejection at every draft position, plus the all-accepted case: the verifier must stop
    // at the first rejection.
    for (int reject_at = 0; reject_at <= 6; ++reject_at) {
        for (bool eos : {false, true}) {
            std::array<llama_token_data, 7> data;
            std::array<llama_token_data_array, 7> rows;
            for (size_t i = 0; i < rows.size(); ++i) {
                const llama_token t = i == (size_t) reject_at ? (eos ? 99 : 98) : (i < draft.size() ? draft[i] : 97);
                data[i] = {t, 0.0f, 1.0f};
                rows[i] = {&data[i], 1, 0, true};
            }
            std::mt19937 rng(321);
            llama_tokens sampler_history;
            size_t sampled = 0;
            const auto accepted = server_speculative_rejection(draft, q, rng,
                    [&](size_t i) { ++sampled; return server_speculative_sample{data[i].id, &rows[i]}; },
                    [&](llama_token t) { sampler_history.push_back(t); },
                    [](llama_token t) { return t == 99; });
            assert(accepted.size() == (size_t) reject_at + 1);
            assert(sampled == accepted.size());
            assert(sampler_history == accepted);
            assert(std::equal(accepted.begin(), accepted.end() - 1, draft.begin()));
        }
    }

    // An accepted EOG proposal stops verification too, not only an EOG residual/replacement.
    std::vector<llama_token_data> p_eog{{99,0.0f,1.0f}};
    llama_token_data_array row_eog{p_eog.data(),p_eog.size(),0,true};
    std::mt19937 rng(123);
    size_t calls = 0;
    auto eog = server_speculative_rejection({99,12}, {{{99,0.0f,1.0f}},{{12,0.0f,1.0f}}}, rng,
            [&](size_t) { ++calls; return server_speculative_sample{99,&row_eog}; },
            [](llama_token) {}, [](llama_token t) { return t == 99; });
    assert(eog == llama_tokens({99}) && calls == 1);
}

static void test_distribution() {
    // First position: a sampled MTP q != p. Both emitted tokens, including EOS, must follow p.
    std::mt19937 proposals(42), verify_rng(43);
    std::uniform_real_distribution<float> uniform(0,1);
    std::vector<llama_token_data> p{{0,0,0.2f},{1,0,0.3f},{2,0,0.5f}};
    std::vector<llama_token_data> q{{0,0,0.6f},{1,0,0.3f},{2,0,0.1f}};
    llama_token_data_array row{p.data(),p.size(),0,true};
    std::array<int,3> count{};
    constexpr int trials = 200000;
    for (int i = 0; i < trials; ++i) {
        const float u = uniform(proposals);
        const llama_token proposal = u < 0.6f ? 0 : (u < 0.9f ? 1 : 2);
        const auto tokens = server_speculative_rejection({proposal}, {q}, verify_rng,
                [&](size_t) { return server_speculative_sample{0,&row}; },
                [](llama_token) {}, [](llama_token t) { return t == 2; });
        ++count[tokens.front()];
    }
    for (size_t i = 0; i < count.size(); ++i) {
        assert(std::fabs((double) count[i]/trials - p[i].p) < 0.005);
    }

    // Depth 4: three accepted MTP positions followed by a deterministic lookup. Its q is a
    // point mass; accepting/rejecting it must still reproduce the target, including EOS.
    llama_token_data one{7,0,1};
    llama_token_data_array certain{&one,1,0,true};
    count = {};
    for (int i = 0; i < trials; ++i) {
        const auto tokens = server_speculative_rejection({7,7,7,1},
                {{{7,0,1}},{{7,0,1}},{{7,0,1}},{{1,0,1}}}, verify_rng,
                [&](size_t pos) { return pos < 3 ? server_speculative_sample{7,&certain} : server_speculative_sample{0,&row}; },
                [](llama_token) {}, [](llama_token t) { return t == 2; });
        assert(tokens.size() >= 4);
        ++count[tokens[3]];
    }
    for (size_t i = 0; i < count.size(); ++i) {
        assert(std::fabs((double) count[i]/trials - p[i].p) < 0.005);
    }
}

int main() {
    test_rejection();
    test_distribution();
    puts("speculative rejection: PASS (every rejection depth, EOG, 400k distribution trials)");
}
