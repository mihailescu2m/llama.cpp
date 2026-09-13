// Repeated TOP_K sets must be reproducible for tied QSA scores. Output order
// need not be sorted, and ties need not match a different backend's convention.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) { puts("SKIP: no GPU backend"); return 0; }
    if (!std::strstr(ggml_backend_name(backend), "MTL") && !std::strstr(ggml_backend_name(backend), "Metal")) {
        ggml_backend_free(backend);
        puts("SKIP: Metal-specific reproducibility test");
        return 0;
    }
    int failures = 0;
    struct shape { int cols, rows, k; };
    for (const auto & shape : {shape{4096,16,257}, shape{8193,3,2051}, shape{33024,1,2051},
                               shape{4096,2,4096}, shape{8193,8,1}}) {
      const int cols = shape.cols, rows = shape.rows, k = shape.k;
      for (int kind = 0; kind < 4; ++kind) {
        auto ctx = ggml_init({ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true});
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        ggml_set_input(input);
        auto * output = ggml_top_k(ctx, input, k);
        ggml_set_output(output);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, graph)) { return 2; }
        std::vector<float> values(cols*rows, 0);
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < cols; ++i) {
                values[r*cols+i] = kind == 3 ? (float) (cols-i) : kind == 0 ? 0.0f : i < 32 ? (float) (32-i) :
                                  kind == 1 ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
        std::vector<int32_t> first, result(rows*k);
        int changed = 0;
        for (int rep = 0; rep < 32; ++rep) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return 2; }
            ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(int32_t));
            for (int r = 0; r < rows; ++r) {
                auto begin = result.begin()+r*k, end = begin+k;
                std::sort(begin, end);
                if (*begin < 0 || *(end-1) >= cols || std::adjacent_find(begin, end) != end) { return 2; }
                for (int i = 0; i < k; ++i) { if (begin[i] != i) { return 2; } }
            }
            if (rep == 0) { first = result; }
            else { changed += result != first; }
        }
        printf("cols=%d rows=%d k=%d kind=%d changed_sets=%d/31\n", cols, rows, k, kind, changed);
        failures += changed > 0;
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
      }
    }
    ggml_backend_free(backend);
    return failures ? 1 : 0;
}
