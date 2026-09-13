// test-top-k-determinism.cpp: GGML_OP_TOP_K must select the same members on every run.
//
// Metal's radix select (rows over 2048 columns with k over 64) used to let atomic arrival order
// decide which members of a cutoff tie survived, so identical inputs produced different selections
// from run to run. Qwen's QSA block indexer takes that path past 2048 blocks, and the selection
// becomes an attention mask - so the logits changed between processes on identical input.
//
// test-backend-ops cannot see this: with ties it deliberately accepts any tie choice. Here the
// contract is exact - everything above the cutoff, then the lowest-index members of the tie - and
// it has to hold on every one of repeated runs. Output order is not part of the contract.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

struct topk_case {
    const char * name;
    int ncols;
    int nrows;
    int k;
    std::vector<float> data;
};

static std::vector<int32_t> run_once(ggml_backend_t backend, const topk_case & c) {
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ncols, c.nrows);
    ggml_set_input(a);
    ggml_tensor * out = ggml_top_k(ctx, a, c.k);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<int32_t> res;
    if (ggml_gallocr_alloc_graph(alloc, gf)) {
        ggml_backend_tensor_set(a, c.data.data(), 0, c.data.size()*sizeof(float));
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            res.resize((size_t) c.k * c.nrows);
            ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(int32_t));
        }
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return res;
}

// everything strictly above the k-th value, then ties in increasing index order
static std::vector<int32_t> expected_row(const float * row, int ncols, int k) {
    std::vector<int32_t> idx(ncols);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t x, int32_t y) { return row[x] > row[y]; });
    idx.resize(k);
    std::sort(idx.begin(), idx.end());
    return idx;
}

static uint32_t lcg(uint32_t & s) { s = s*1664525u + 1013904223u; return s >> 8; }

int main() {
    std::vector<topk_case> cases;

    {   // every column tied: the whole selection is decided by the tie rule
        topk_case c = { "all tied", 4096, 2, 257, {} };
        c.data.assign((size_t) c.ncols*c.nrows, 1.0f);
        cases.push_back(c);
    }
    {   // 200 distinct winners, then a 300-wide tie straddling the cutoff, then distinct losers
        topk_case c = { "cutoff tie", 4096, 3, 257, {} };
        c.data.resize((size_t) c.ncols*c.nrows);
        uint32_t s = 12345;
        for (int r = 0; r < c.nrows; ++r) {
            float * row = c.data.data() + (size_t) r*c.ncols;
            for (int i = 0; i < c.ncols; ++i) {
                row[i] = 1.0f + 1e-4f*(float) i/c.ncols; // distinct, below the tie
            }
            std::vector<int> pos(c.ncols);
            std::iota(pos.begin(), pos.end(), 0);
            for (int i = c.ncols - 1; i > 0; --i) { std::swap(pos[i], pos[lcg(s) % (i + 1)]); }
            for (int i = 0; i < 200; ++i)       { row[pos[i]] = 10.0f + 1e-3f*i; }
            for (int i = 200; i < 500; ++i)     { row[pos[i]] = 5.0f; }
        }
        cases.push_back(c);
    }
    {   // no ties: the unchanged fast path
        topk_case c = { "distinct", 4096, 2, 257, {} };
        c.data.resize((size_t) c.ncols*c.nrows);
        for (size_t i = 0; i < c.data.size(); ++i) { c.data[i] = (float) ((i*2654435761u) % 1000003u); }
        cases.push_back(c);
    }

    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            backend = ggml_backend_dev_init(dev, nullptr);
            printf("device: %s\n", ggml_backend_dev_name(dev));
            break;
        }
    }
    if (backend == nullptr) {
        printf("no GPU backend - skipping\n");
        return 0;
    }

    const int n_runs = 16;
    int n_fail = 0;
    for (const auto & c : cases) {
        int bad_runs = 0;
        for (int run = 0; run < n_runs; ++run) {
            const std::vector<int32_t> got = run_once(backend, c);
            if (got.empty()) { bad_runs++; continue; }
            for (int r = 0; r < c.nrows; ++r) {
                std::vector<int32_t> sel(got.begin() + (size_t) r*c.k, got.begin() + (size_t) (r + 1)*c.k);
                std::sort(sel.begin(), sel.end());
                if (sel != expected_row(c.data.data() + (size_t) r*c.ncols, c.ncols, c.k)) {
                    bad_runs++;
                    break;
                }
            }
        }
        printf("%-12s ncols=%d k=%d: %s (%d/%d runs matched the contract)\n",
                c.name, c.ncols, c.k, bad_runs == 0 ? "ok" : "FAIL", n_runs - bad_runs, n_runs);
        n_fail += bad_runs != 0;
    }

    ggml_backend_free(backend);
    printf("top_k determinism: %s\n", n_fail == 0 ? "PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
