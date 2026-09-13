#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

// Shared framing for checkpoint boundary rows. A type tag is needed because checkpoint restore
// is broadcast to the speculative implementations; size alone cannot establish ownership.
namespace common_speculative_state {

constexpr uint32_t magic_mtp    = 0x4250544dU; // "MTPB"
constexpr uint32_t magic_eagle3 = 0x42334745U; // "EG3B"
constexpr uint32_t version     = 1;

struct boundary_header {
    uint32_t  magic   = magic_mtp;
    uint32_t  version = common_speculative_state::version;
    int32_t   n_embd  = 0;
    int32_t   pad     = 0;
    llama_pos pos     = -1;
    int32_t   pad2    = 0;
};
static_assert(sizeof(boundary_header) == 24, "boundary format must have a stable layout");

enum class boundary_result { valid, foreign, invalid };

inline boundary_result read_boundary(const uint8_t * data, size_t size, uint32_t magic,
                                     int32_t n_embd, llama_pos next_pos, boundary_header & hdr) {
    if (data == nullptr || size < sizeof(hdr)) {
        return boundary_result::invalid;
    }
    std::memcpy(&hdr, data, sizeof(hdr));

    // Only a recognized, complete foreign record is evidence that this is somebody else's
    // state. Unknown/corrupt magic, old headerless Eagle3 blobs and truncated rows fail closed.
    if ((hdr.magic != magic_mtp && hdr.magic != magic_eagle3) || hdr.version != version ||
        hdr.n_embd <= 0 || hdr.pos < 0 ||
        (size - sizeof(hdr)) % sizeof(float) != 0 ||
        (size - sizeof(hdr)) / sizeof(float) != (size_t) hdr.n_embd) {
        return boundary_result::invalid;
    }
    if (hdr.magic != magic) {
        return boundary_result::foreign;
    }
    // Subtract from the checked expected position, not an untrusted hdr.pos + 1 (which can overflow).
    if (hdr.n_embd != n_embd || (next_pos >= 0 && hdr.pos != next_pos - 1)) {
        return boundary_result::invalid;
    }
    return boundary_result::valid;
}

} // namespace common_speculative_state
