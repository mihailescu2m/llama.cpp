// Model-input integration checks, without loading model weights or submitting GPU work.
#include "../src/llama-model.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-batch.h"
#include "../src/llama-moe-stream.h"
#include "../src/llama-attention-union.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) { throw std::runtime_error("check failed: " #x); } } while (0)

static void test_qsa_tables() {
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    auto & hp = model->hparams;
    hp.n_layer_all = 1; hp.n_embd = 32;
    hp.n_head_arr[0] = hp.n_head_kv_arr[0] = 1;
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
    hp.indexer_head_size = 32; hp.dsv4_compress_ratios[0] = 4;
    llama_memory_hybrid_idx mem(*model,
        GGML_TYPE_F32, GGML_TYPE_F32, false, 16, 1, 0, LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F32, GGML_TYPE_F32, 2, 2, 0, false, true,
        [](int32_t) {return true;}, [](int32_t) {return false;}, [](int32_t) {return true;});
    auto & cells = const_cast<llama_kv_cells &>(mem.get_mem_idx()->get_cells(0));
    ggml_context_ptr ctx{ggml_init({ggml_tensor_overhead()*8, nullptr, true})};
    auto * blocks = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 16, 1);
    auto * bias = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 1, 1);
    auto * dc = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 16, 1);
    auto * dp = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 16);
    auto * dr = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 4);
    ggml_backend_buffer_ptr buf{ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())};
    CHECK(buf);
    llama_pos pos = 11;
    llama_seq_id seq = 0;
    llama_seq_id * seqp = &seq;
    int32_t ns = 1;
    llama_ubatch ub{};
    ub.n_tokens = ub.n_seq_tokens = ub.n_seqs = ub.n_seqs_unq = ub.n_pos = 1;
    ub.pos = &pos; ub.seq_id = &seqp; ub.n_seq_id = &ns;
    llama_memory_hybrid_idx_context mctx(&mem);

    // The block-selection table has only one spare block. Two partial sequences
    // or two partial position blocks must use per-cell selection, not overwrite
    // each other's live keys in that spare block.
    cells.reset();
    for (int i = 0; i < 7; ++i) { cells.pos_set(i, i); cells.seq_add(i, 0); }
    CHECK(mctx.qsa_block_topk_safe(ub));
    cells.seq_rm(1, 0); // remove position 1 without removing a whole block
    CHECK(!mctx.qsa_block_topk_safe(ub));
    cells.reset();
    for (int i = 0; i < 6; ++i) {
        cells.pos_set(i, i % 3); cells.seq_add(i, i / 3);
    }
    CHECK(!mctx.qsa_block_topk_safe(ub));

    // Contiguous, hole, unified sequences, prefix removal; include cached decode steps.
    for (int scenario = 0; scenario < 4; ++scenario) {
        cells.reset(); mem.pooled_valid(0) = 0; mem.pooled_valid(1) = 0;
        for (int i = 0; i < (scenario == 3 ? 12 : 8); ++i) {
            cells.pos_set(i, scenario == 2 ? i%4 : (scenario == 1 && i >= 4 ? i+4 : i));
            cells.seq_add(i, scenario == 2 ? i/4 : 0);
        }
        pos = scenario == 2 ? 3 : scenario == 0 ? 7 : 11;
        float cache[6] = {NAN, NAN, NAN, NAN, NAN, NAN};
        for (int step = 0; step < 3; ++step) {
            if (scenario == 3 && step == 1) { CHECK(mem.seq_rm(0, 0, 4)); }
            const auto needed = mctx.qsa_pooled_n_dirty_max(ub, 4);
            CHECK(needed <= 4 && (scenario != 2 || needed >= 2));
            mem.set_input_qsa(nullptr, blocks, nullptr, bias, &ub, 4, 16, true, dc, dp, dr);
            const auto * b = (int32_t *) blocks->data;
            const auto * d = (int32_t *) dc->data;
            const auto * r = (int64_t *) dr->data;
            for (int i = 0; i < 4; ++i) {
                CHECK(r[i] >= 0 && r[i] < 6);
                cache[r[i]] = 0;
                for (int j = 0; j < 4; ++j) { cache[r[i]] += d[4*i+j] + 1; }
            }
            for (int block = 0; block < (scenario == 3 && step == 0 ? 3 : 2); ++block) {
                float expected = 0;
                for (int j = 0; j < 4; ++j) { expected += b[4*block+j] + 1; }
                CHECK(cache[block] == expected);
            }
        }
    }
}

static void test_union_causality_and_support() {
    ggml_context_ptr ctx{ggml_init({1 << 20, nullptr, false})};
    auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, 1);
    auto * kv = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 1, 5, 1);
    auto * scores = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 5, 8);
    for (int t = 0; t < 8; ++t) {
        ((float *) q->data)[t] = 0;
        for (int c = 0; c < 4; ++c) { ((float *) scores->data)[t*4+c] = c == 0 ? 0.0f : -INFINITY; }
        for (int c = 0; c < 5; ++c) { ((ggml_fp16_t *) mask->data)[t*5+c] = ggml_fp32_to_fp16(c < 2 ? 0.0f : -INFINITY); }
    }
    for (int c = 0; c < 5; ++c) { ((ggml_fp16_t *) kv->data)[c] = ggml_fp32_to_fp16(c < 2 ? 0.0f : 100.0f); }
    auto * top = ggml_cont(ctx.get(), ggml_top_k(ctx.get(), scores, 2));
    auto * uids = ggml_union_build(ctx.get(), top, 4, 8);
    auto * out = llama_attention_union_masked(ctx.get(), q, kv, kv, mask, uids, 1, 1.0f);
    auto * raw_only = ggml_flash_attn_union(ctx.get(), q, kv, kv, mask, uids, 1, 1.0f);
    auto * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, raw_only);
    CHECK(ggml_graph_compute_with_ctx(ctx.get(), gf, 1) == GGML_STATUS_SUCCESS);
    for (int t = 0; t < 8; ++t) {
        CHECK(((float *) out->data)[t] == 0);
        CHECK(((float *) raw_only->data)[t] > 0); // old masking demonstrably admits a future key
    }

    ggml_backend_device dev{};
    dev.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
    dev.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
    CHECK(llama_attention_union_supported(&dev, uids, out));
    dev.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * op) { return op->op != GGML_OP_UNION_BUILD; };
    CHECK(!llama_attention_union_supported(&dev, uids, out));
    dev.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor * op) { return op->op != GGML_OP_FLASH_ATTN_UNION; };
    CHECK(!llama_attention_union_supported(&dev, uids, out));
    dev.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; };
    dev.iface.supports_op = [](ggml_backend_dev_t, const ggml_tensor *) { return true; };
    CHECK(!llama_attention_union_supported(&dev, uids, out));
    CHECK(!llama_attention_union_supported(nullptr, uids, out));
}

static void test_pair_padding_and_host_pointer() {
    llama_moe_stream mgr(1, 4, 1, false);
    ggml_context_ptr ctx{ggml_init({ggml_tensor_overhead()*2, nullptr, true})};
    auto * original = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 32, 1, 16);
    mgr.create_cache_tensor(0, ggml_backend_cpu_buffer_type(), original, 0, 0);
    mgr.alloc_bufs(false);
    auto & sl = *mgr.layers[0];
    sl.slot_state[0] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
    sl.slot_expert[0] = 0; sl.expert_slot[0] = 0; sl.plan_pool = {0};
    std::vector<int32_t> ids(32, 0), out(32*LLAMA_MOE_PAIR_ROWS);
    for (int skip : {0, 1}) {
        mgr.pad_skip = skip;
        for (bool empty : {false, true}) {
            sl.plan_pair = {empty ? std::vector<int32_t>{} : std::vector<int32_t>{7}};
            mgr.emit_wave_pairs(sl, ids.data(), out.data(), 0, 1, 32, 32);
            for (int i = 0; i < 32; ++i) {
                CHECK(out[LLAMA_MOE_PAIR_INPUT*32+i] >= 0 && out[LLAMA_MOE_PAIR_INPUT*32+i] < 32);
                CHECK(out[LLAMA_MOE_PAIR_PAIR*32+i] <= 32);
            }
            CHECK(out[LLAMA_MOE_PAIR_INPUT*32] == (empty ? 0 : 7));
        }
    }
    const auto * cache = sl.weights[0].cache;
    CHECK(ggml_backend_tensor_get_host_ptr(cache) == cache->data);
    CHECK(ggml_backend_tensor_get_host_ptr(nullptr) == nullptr);
}

int main() {
    try {
        test_qsa_tables(); test_union_causality_and_support(); test_pair_padding_and_host_pointer();
        puts("fork attention / padding / host access: PASS");
        return 0;
    } catch (const std::exception & err) { fprintf(stderr, "%s\n", err.what()); return 1; }
}
