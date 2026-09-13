// Note: this description is outdated
//
// An interface allowing to compute ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using ggml_graph_compute(), you
// use ggml_metal_graph_compute() (or ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the ggml_metal_set_tensor() and ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

// TODO: remove in the future
GGML_BACKEND_API ggml_backend_t ggml_backend_metal_init(void);

GGML_BACKEND_API bool ggml_backend_is_metal(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
GGML_BACKEND_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);

// capture all command buffers committed the next time `ggml_backend_graph_compute` is called
GGML_BACKEND_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);

// Replace the GPU views of a shared (host-memory) buffer by n page-aligned [offs[i], offs[i] + sizes[i])
// ranges, ascending, non-overlapping and inside the buffer. Tensors must lie entirely within one view.
// The new views are registered for residency before the old ones are dropped, so memory both cover
// stays wired throughout and only what the new views leave out loses its wiring - which the driver
// does asynchronously. A caller returning that memory to the system must wait for it: a page released
// while still wired stays with the allocation, unmapped and unreachable, until the system swaps it.
// Returns false and leaves the buffer unchanged if it is not a Metal shared buffer or a view cannot
// be created. The device must be idle.
GGML_BACKEND_API bool ggml_backend_metal_buffer_set_views(ggml_backend_buffer_t buffer, const size_t * offs, const size_t * sizes, int n);

// Allocate a buffer of the Metal shared buffer type whose n page-aligned [offs[i], offs[i] + sizes[i])
// ranges are separate memory objects, so each can later be returned to the system whole: the kernel does
// not free part of an object while the rest stays mapped. Returns NULL if buft is not that buffer type.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_split(ggml_backend_buffer_type_t buft, size_t size,
                                                                                 const size_t * offs, const size_t * sizes, int n);

// whether ggml_backend_metal_buffer_set_views applies to this buffer
GGML_BACKEND_API bool ggml_backend_metal_buffer_has_views(ggml_backend_buffer_t buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_metal_reg(void);

#ifdef __cplusplus
}
#endif
