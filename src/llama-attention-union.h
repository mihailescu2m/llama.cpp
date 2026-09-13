#pragma once

#include "ggml-backend.h"

// Top-k padding can include invisible compressed keys. Apply the original mask to
// every selected key, not just the raw window, in both gathered-attention graphs.
inline ggml_tensor * llama_attention_union_masked(ggml_context * ctx,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
        ggml_tensor * ids, int64_t n_raw, float scale) {
    auto * out = ggml_flash_attn_union(ctx, q, k, v, mask, ids, n_raw, scale);
    ggml_flash_attn_union_mask_all(out);
    return out;
}

// Do not replace native attention with the single-thread CPU reference implementation.
// Query the layer's actual device, including shape support, rather than its backend name.
inline bool llama_attention_union_supported(ggml_backend_dev_t dev,
        const ggml_tensor * ids, const ggml_tensor * attention) {
    return dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
           ggml_backend_dev_supports_op(dev, ids) && ggml_backend_dev_supports_op(dev, attention);
}
