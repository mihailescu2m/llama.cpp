#pragma once

#include "ggml-backend.h"

#include <cstring>

// Only count workspace allocations from the same physical memory pool as the host
// expert cache. Inspect the scheduler's buffer type, not the model's device list:
// auxiliary backends such as Accelerate can also own compute allocations.
inline bool llama_memory_phase_buft_uses_system_ram(ggml_backend_buffer_type_t buft) {
    if (!buft) {
        return false;
    }
    if (ggml_backend_buft_is_host(buft)) {
        return true;
    }
#if defined(__APPLE__) && defined(__aarch64__)
    // Apple Silicon Metal buffers use unified RAM, including private buffers.
    // is_host is deliberately false to prevent CPU scheduling on Metal buffers.
    // This fork's registry name is MTL; older versions used Metal. Do not extend
    // this exception to discrete Metal GPUs on Intel Macs.
    const auto dev = ggml_backend_buft_get_device(buft);
    const auto reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name && (std::strcmp(name, "MTL") == 0 || std::strcmp(name, "Metal") == 0);
#else
    return false;
#endif
}
