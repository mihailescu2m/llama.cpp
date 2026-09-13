#include "../src/llama-moe-wave.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>

int main() {
    assert(llama_moe_stream_default_slots(512, 10) == 30);
    assert(llama_moe_stream_default_slots(256, 6) == 18);
    assert(llama_moe_stream_default_slots(8, 2) == 8);
    assert(llama_moe_stream_wave_cap(30, 10) == 10);
    assert(llama_moe_stream_wave_cap(20, 10) == 0);
    assert(llama_moe_stream_wave_cap(20, 10, 100) == 10);
    assert(llama_moe_stream_wave_cap(9, 10, 100) == 0);
    // The reported regression: a legal cap of 80 used to be shrunk to 52, adding a masked pass.
    assert(llama_moe_stream_partition_cap(256, 110, 6, 1, 80) == 80);
    // Preserve the intended optimization: 512 tokens can support five partitioned waves.
    assert(llama_moe_stream_partition_cap(256, 110, 6, 5, 27) == 52);
    // If partitioning is unreachable, a forced measurement cap must remain unchanged.
    assert(llama_moe_stream_partition_cap(256, 110, 6, 1, 27) == 27);

    size_t cases = 0;
    for (uint32_t slots = 18; slots < 256; ++slots) {
        for (uint32_t tokens : { 4, 32, 132, 256, 512, 1024, 2048 }) {
            const uint64_t touch = std::min<uint64_t>(256, tokens * 6);
            if (touch <= slots) { continue; }
            const uint32_t max_waves = std::max(1u, tokens * 6 / 600);
            for (uint32_t cap = 6; cap <= slots - 6; ++cap) {
                const uint64_t old_waves = (touch + cap - 1) / cap;
                const uint32_t next = old_waves > max_waves
                    ? llama_moe_stream_partition_cap(touch, slots, 6, max_waves, cap) : cap;
                const uint64_t waves = (touch + next - 1) / next;
                assert(next >= cap);
                assert(waves <= old_waves);
                if (next != cap) {
                    assert(waves <= max_waves);
                    assert(next <= (slots - 6) / 2);
                }
                ++cases;
            }
        }
    }
    printf("MoE wave planning: PASS (%zu configurations)\n", cases);
}
