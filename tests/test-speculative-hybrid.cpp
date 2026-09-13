#ifdef NDEBUG
#undef NDEBUG
#endif

#include "common.h"
#include "speculative.h"
#include "speculative-ngram.h"
#include "../tools/server/server-speculative.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

static void test_limits() {
    common_params_speculative p;
    p.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    p.draft.n_max = 3;
    assert(p.need_n_rs_seq() == 3);
    assert(common_speculative_n_max(&p) == 3);
    p.mtp_ngram_n_max = 3;
    assert(p.need_n_rs_seq() == 3); // K stays 4, not 7
    assert(common_speculative_n_max(&p) == 6);
    const auto limits = common_speculative_get_output_limits(512, 1, common_speculative_n_max(&p));
    assert(limits.total >= 7 && limits.per_seq >= 7);
    p.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
    p.ngram_mod.n_max = 16;
    assert(p.need_n_rs_seq() == 3);
    assert(common_speculative_n_max(&p) == 16); // fallback still independent of composition

    auto invalid = [&](const common_params_speculative & q) {
        bool threw = false;
        try { common_speculative_n_max(&q); } catch (const std::invalid_argument &) { threw = true; }
        assert(threw);
    };
    auto q = p; q.types = {COMMON_SPECULATIVE_TYPE_NGRAM_MOD}; invalid(q);
    q = p; q.draft.n_max = 0; invalid(q);
    q = p; q.mtp_ngram_n_max = -1; invalid(q);
    q = p; q.mtp_ngram_n_max = 65; invalid(q);
    q = p; q.ngram_mod.n_match = 0; invalid(q);
    q = p; q.ngram_mod.n_match = 65536; invalid(q);
    q = p; q.draft.n_max = INT32_MAX; invalid(q);
}

static void test_batch_budget() {
    // -b 4 fits only anchor + MTP 3, even when the configured combined draft is six.
    assert(server_speculative_draft_limit(6, 4, 1) == 3);
    assert(server_speculative_draft_limit(6, 7, 1) == 6);
    assert(server_speculative_draft_limit(6, 4096, 1) == 6);
    assert(server_speculative_draft_limit(2, 7, 1) == 2); // output/context cap still wins
    assert(server_speculative_draft_limit(0, 7, 1) == 0);
    assert(server_speculative_draft_limit(-1, 7, 1) == 0); // exhausted context, not unlimited
    assert(server_speculative_draft_limit(6, 1, 1) == 0); // anchor only, do not call draft()
    assert(server_speculative_draft_limit(6, 0, 1) == 0);
    assert(server_speculative_draft_limit(6, 7, 0) == 0);

    // Account for other slots and non-divisible capacities, not n_batch independently per slot.
    assert(server_speculative_draft_limit(6, 8, 2) == 3);
    assert(server_speculative_draft_limit(6, 7, 2) == 2);
    assert(server_speculative_draft_limit(6, 3, 4) == 0);
    assert(server_speculative_draft_limit(INT32_MAX, UINT32_MAX, 1) == INT32_MAX);

    // A real lookup hit must be suppressed when the model prefix exhausts the batch allowance.
    common_speculative_ngram lookup(4);
    const llama_tokens history{1,2,3,4,5,6,7,8,999};
    const llama_tokens prefix{2,3,4};
    for (const uint32_t n_batch : {4, 5, 6, 7}) {
        const int n_max = server_speculative_draft_limit(6, n_batch, 1);
        const auto tail = lookup.draft(history, 1, prefix, n_max - (int) prefix.size());
        assert(tail.size() == n_batch - 4);
        assert(1 + prefix.size() + tail.size() <= n_batch);
    }

    // Replay decisions were made under the same fixed allowance. Even if all idle slots join
    // between verification and replay, anchors + every retained decision still fit together.
    for (uint32_t n_batch = 1; n_batch <= 64; ++n_batch) {
        for (uint32_t n_parallel = 1; n_parallel <= 16; ++n_parallel) {
            const auto n_max = server_speculative_draft_limit(6, n_batch, n_parallel);
            if (n_parallel > n_batch) {
                assert(n_max == 0); // no speculative space beyond the anchors
                continue;
            }
            assert(n_parallel * (1 + n_max) <= n_batch);
            for (int n_decided = 1; n_decided <= n_max; ++n_decided) {
                const auto remaining_slots = (n_parallel - 1) * (1 + n_max);
                assert(1 + n_decided + remaining_slots <= n_batch);
            }
        }
    }
}

static void test_lookup() {
    common_speculative_ngram lookup(4);
    assert(lookup.size_bytes() == 1024*1024);
    llama_tokens history{1,2,3,4,5,6,7,8,999};
    const llama_tokens prefix{2,3,4};
    assert(lookup.draft(history, 1, prefix, 3) == llama_tokens({5,6,7}));
    assert(lookup.draft(history, 1, prefix, 1) == llama_tokens({5}));
    assert(lookup.draft(history, 1, prefix, 0).empty());
    assert(lookup.draft(history, 1, {2,3,77}, 3).empty());

    // A collision must be a miss, never a guessed successor from an unrelated n-gram.
    common_speculative_ngram colliding(4, 1);
    assert(colliding.draft(history, 1, prefix, 3).empty());

    // Even after an in-place history edit, values are read from the current verified history.
    history[4] = 55;
    assert(lookup.draft(history, 1, prefix, 1) == llama_tokens({55}));
    lookup.reset();
    assert(lookup.draft(history, 1, prefix, 1) == llama_tokens({55}));
    assert(lookup.draft({98,97}, 1, prefix, 3).empty()); // rewind rebuilds the index

    common_speculative_ngram other_slot(4);
    assert(other_slot.draft({98,97,96,95,94}, 1, prefix, 3).empty());
    assert(other_slot.draft({}, 1, {}, 3).empty());

    // A hypothetical MTP prefix cannot teach the lookup its own continuation.
    common_speculative_ngram clean(2);
    assert(clean.draft({90,91,92}, 1, {2,3,4}, 3).empty());
    assert(clean.draft({90,91,92}, 1, {2}, 3).empty());

    // Incrementally grow / replace / rewind history. Every proposed token must have an exact,
    // fully known supporting occurrence; the generated suffix is never its own evidence.
    std::mt19937 rng(912);
    for (size_t match = 1; match <= 8; ++match) {
        common_speculative_ngram index(match, 32);
        llama_tokens past;
        for (int trial = 0; trial < 500; ++trial) {
            if (trial % 29 == 0) { past.clear(); }
            past.push_back(rng() % 7);
            const llama_token anchor = rng() % 7;
            const llama_tokens mtp{(llama_token) (rng()%7), (llama_token) (rng()%7), (llama_token) (rng()%7)};
            const auto tail = index.draft(past, anchor, mtp, 5);
            auto verified = past;
            verified.push_back(anchor);
            auto virtual_prefix = verified;
            virtual_prefix.insert(virtual_prefix.end(), mtp.begin(), mtp.end());
            for (auto token : tail) {
                bool found = false;
                for (size_t pos = 0; pos + match < verified.size(); ++pos) {
                    if (verified[pos + match] == token && std::equal(
                            verified.begin() + pos, verified.begin() + pos + match,
                            virtual_prefix.end() - match)) {
                        found = true;
                        break;
                    }
                }
                assert(found);
                virtual_prefix.push_back(token);
            }
        }
    }
}

static void test_rejection_and_replay() {
    const llama_tokens draft{10,11,12,13,14,15};
    std::vector<std::vector<llama_token_data>> q;
    for (auto t : draft) { q.push_back({{t, 0.0f, 1.0f}}); }

    // Force rejection at every MTP/suffix position, plus the all-accepted case. The verifier
    // must stop at the first rejection and checkpoint replay must never resample that decision.
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

            server_speculative_replay replay;
            replay.start(accepted);
            assert(replay.n_accepted == (size_t) reject_at);
            const auto rng_post_decision = rng;
            size_t draws = 0;
            const auto result = replay.finish(accepted,
                    [&]() { ++draws; return 77; },
                    [&](llama_token t) { sampler_history.push_back(t); },
                    [](llama_token t) { return t == 99; });
            assert(rng == rng_post_decision);
            assert(std::equal(accepted.begin(), accepted.end(), result.begin()));
            assert(result == sampler_history); // no duplicate sampler accepts
            assert(draws == (eos ? 0 : 1));
            assert(result.size() == accepted.size() + draws);
            // Replay evaluates anchor + decided; stopping at EOS needs exactly one rollback.
            assert(accepted.size() + 1 - result.size() <= 1);
            replay.clear();
            assert(!replay.active && replay.n_accepted == 0);
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
    test_limits();
    test_batch_budget();
    test_lookup();
    test_rejection_and_replay();
    test_distribution();
    puts("MTP + n-gram: PASS (K=4 limits, multi-slot batch budget, bounded lookup, collision/rewind/isolation, every rejection depth, replay/EOG, 400k distribution trials)");
}
