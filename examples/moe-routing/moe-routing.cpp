// Capture per-layer MoE routing state during a normal decode, for offline analysis of whether
// layer L's activations predict layer L+1's expert selection.
//
// Records four tensors per layer per token: ffn_norm (router input), ffn_moe_logits (router
// scores), ffn_moe_out (routed contribution), ffn_shexp (shared contribution). Output is a flat
// binary stream, see the record header below.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

static const char * const k_tags[] = { "ffn_norm", "ffn_moe_logits", "ffn_moe_out", "ffn_shexp" };
static const int k_n_tags = 4;

struct capture_data {
    FILE * out = nullptr;
    uint32_t tag_mask = 0xf;   // MOE_ROUTING_TAGS bitmask; logits-only keeps a long prefill small
    int    step = 0;   // decode step; prompt is step 0, each generated token increments it
    int64_t n_rec = 0;
    std::vector<uint8_t> buf;
};

// name must be exactly "<tag>-<il>". ggml views keep the source name with a suffix, e.g.
// "ffn_norm-5 (reshaped)", and those are duplicates of data already captured
static int tag_of(const char * name, int * il) {
    const char * dash = strrchr(name, '-');
    if (!dash || dash[1] == '\0') {
        return -1;
    }
    for (const char * p = dash + 1; *p; p++) {
        if (*p < '0' || *p > '9') {
            return -1;
        }
    }
    for (int t = 0; t < k_n_tags; t++) {
        const size_t len = strlen(k_tags[t]);
        if ((size_t)(dash - name) == len && strncmp(name, k_tags[t], len) == 0) {
            *il = atoi(dash + 1);
            return t;
        }
    }
    return -1;
}

static bool cb_capture(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cd = (capture_data *) user_data;

    int il = 0;
    const int tag = tag_of(t->name, &il);

    if (ask) {
        return tag >= 0 && (cd->tag_mask & (1u << tag));
    }
    if (tag < 0 || !(cd->tag_mask & (1u << tag))) {
        return true;
    }

    // only f32 reaches here for these four tensors; anything else means the graph changed
    if (t->type != GGML_TYPE_F32) {
        LOG_ERR("%s: %s is %s, expected f32\n", __func__, t->name, ggml_type_name(t->type));
        return true;
    }

    // a strided view would make the flat dump unreadable offline
    if (!ggml_is_contiguous(t)) {
        LOG_ERR("%s: %s is not contiguous, skipped\n", __func__, t->name);
        return true;
    }

    const size_t nb = ggml_nbytes(t);

    cd->buf.resize(nb);
    ggml_backend_tensor_get(t, cd->buf.data(), 0, nb);

    const int32_t hdr[8] = { (int32_t) tag, (int32_t) il, (int32_t) cd->step,
                             (int32_t) t->ne[0], (int32_t) t->ne[1], (int32_t) t->ne[2], (int32_t) t->ne[3],
                             (int32_t) nb };
    fwrite(hdr, sizeof(hdr), 1, cd->out);
    fwrite(cd->buf.data(), 1, nb, cd->out);
    cd->n_rec++;

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_predict = 128;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // -o is not wired for LLAMA_EXAMPLE_COMMON, so take the path from the environment
    const char * env_out = getenv("MOE_ROUTING_OUT");
    const std::string out_path = env_out ? env_out : "moe-routing.bin";

    capture_data cd;
    if (const char * e = getenv("MOE_ROUTING_TAGS")) {
        cd.tag_mask = (uint32_t) strtoul(e, nullptr, 0);
    }
    cd.out = fopen(out_path.c_str(), "wb");
    if (!cd.out) {
        LOG_ERR("%s: cannot open %s\n", __func__, out_path.c_str());
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // MOE_ROUTING_TAGS=0 leaves cb_eval unset entirely: attaching it splits the graph at every
    // observed node, so a timing run must not install it at all
    if (cd.tag_mask) {
        params.cb_eval = cb_capture;
        params.cb_eval_user_data = &cd;
    }
    params.warmup = false;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> inp = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (inp.empty()) {
        LOG_ERR("%s: empty prompt\n", __func__);
        return 1;
    }
    LOG_INF("%s: prompt = %zu tokens, generating %d\n", __func__, inp.size(), params.n_predict);

    // Prefill in n_batch chunks, like the server does. Steps count down from 0 so decode steps stay
    // positive and the offline pass can tell the two phases apart.
    const int32_t n_batch = (int32_t) llama_n_batch(ctx);
    int chunk = 0;
    for (size_t off = 0; off < inp.size(); off += n_batch, chunk++) {
        const int32_t n = (int32_t) std::min<size_t>(n_batch, inp.size() - off);
        cd.step = -chunk;
        if (llama_decode(ctx, llama_batch_get_one(inp.data() + off, n))) {
            LOG_ERR("%s: prompt decode failed at offset %zu\n", __func__, off);
            return 1;
        }
    }

    // NOT greedy: this model falls into repetition attractors at temp 0, and repeated text would
    // make routing look far more predictable than it is
    common_sampler * smpl = common_sampler_init(model, params.sampling);

    std::vector<llama_token> gen;
    for (int i = 0; i < params.n_predict; i++) {
        cd.step = i + 1;

        llama_token id = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, id, true);

        if (llama_vocab_is_eog(vocab, id)) {
            LOG_INF("%s: eog at step %d\n", __func__, i);
            break;
        }
        gen.push_back(id);

        if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
            LOG_ERR("%s: decode failed at step %d\n", __func__, i);
            break;
        }
    }

    fclose(cd.out);

    // token ids alongside, so the offline pass can identify the hash-routed layers' inputs
    const std::string tok_path = out_path + ".tokens";
    if (FILE * ft = fopen(tok_path.c_str(), "wb")) {
        const int32_t n_prompt = (int32_t) inp.size();
        fwrite(&n_prompt, sizeof(n_prompt), 1, ft);
        fwrite(inp.data(), sizeof(llama_token), inp.size(), ft);
        const int32_t n_gen = (int32_t) gen.size();
        fwrite(&n_gen, sizeof(n_gen), 1, ft);
        fwrite(gen.data(), sizeof(llama_token), gen.size(), ft);
        fclose(ft);
    }

    LOG_INF("%s: wrote %" PRId64 " records to %s (%d generated tokens)\n",
            __func__, cd.n_rec, out_path.c_str(), (int) gen.size());

    common_sampler_free(smpl);
    llama_backend_free();

    return 0;
}
