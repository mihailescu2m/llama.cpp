#pragma once

#include <algorithm>
#include <cstdint>

inline uint32_t llama_moe_stream_default_slots(uint32_t n_expert, uint32_t n_used) {
    return (uint32_t) std::min<uint64_t>(n_expert, std::max<uint64_t>(16, 3ull*n_used));
}

// Zero denotes an unrepresentable wave; never invoke clamp with reversed bounds.
//
// The default cap follows the double-buffered rule (this wave, the next wave's preload and
// n_used parking slots: cap = (n_slots - n_used)/2), so it needs 3*n_used slots. A FORCED cap
// is only bounded by n_slots - n_used and may therefore exceed that rule between 2*n_used and
// 3*n_used slots: it is a measurement override, and stage_wave_locked bounds its keep-set by
// the real slack at run time.
inline uint32_t llama_moe_stream_wave_cap(uint32_t n_slots, uint32_t n_used, uint32_t forced = 0) {
    if (n_used == 0 || n_slots < 2ull*n_used) { return 0; }
    const uint32_t cap = forced ? std::clamp(forced, n_used, n_slots - n_used) : (n_slots - n_used)/2;
    return cap >= n_used ? cap : 0;
}

// Retune a wave cap only if it actually enables pair partitioning. In particular, a valid forced
// cap may exceed the double-buffered default: clamping it DOWN adds masked passes, not partitioning.
inline uint32_t llama_moe_stream_partition_cap(uint64_t n_touch, uint32_t n_slots,
                                              uint32_t n_used, uint32_t max_waves, uint32_t cap) {
    if (max_waves == 0 || n_slots <= n_used || n_used == 0) {
        return cap;
    }
    const uint32_t cap_max = (n_slots - n_used) / 2;
    if (cap_max < n_used) {
        return cap;
    }
    const uint64_t want = (n_touch + max_waves - 1) / max_waves;
    const uint32_t candidate = (uint32_t) std::clamp<uint64_t>(want, n_used, cap_max);
    return candidate > cap && (n_touch + candidate - 1) / candidate <= max_waves ? candidate : cap;
}
