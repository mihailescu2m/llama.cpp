#include "../common/speculative-state.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <limits>
#include <vector>

using namespace common_speculative_state;

static std::vector<uint8_t> make_blob(uint32_t magic, int32_t width = 4) {
    boundary_header hdr;
    hdr.magic = magic;
    hdr.n_embd = width;
    hdr.pos = 7;
    std::vector<uint8_t> data(sizeof(hdr) + width * sizeof(float));
    std::memcpy(data.data(), &hdr, sizeof(hdr));
    return data;
}

int main() {
    for (uint32_t magic : { magic_mtp, magic_eagle3 }) {
        const auto valid = make_blob(magic);
        boundary_header hdr;
        auto read = [&](const std::vector<uint8_t> & data, llama_pos next = -1) {
            return read_boundary(data.data(), data.size(), magic, 4, next, hdr);
        };
        assert(read(valid) == boundary_result::valid);
        assert(hdr.pos == 7);
        assert(read(valid, 8) == boundary_result::valid);
        assert(read(valid, 9) == boundary_result::invalid);
        assert(read(valid, 0) == boundary_result::invalid);
        assert(read({}) == boundary_result::invalid);
        assert(read_boundary(nullptr, valid.size(), magic, 4, -1, hdr) == boundary_result::invalid);

        for (size_t size = 0; size < valid.size(); ++size) {
            auto short_blob = valid;
            short_blob.resize(size);
            assert(read(short_blob) == boundary_result::invalid);
        }
        auto extra = valid;
        extra.push_back(0);
        assert(read(extra) == boundary_result::invalid);
        assert(read(make_blob(magic, 5)) == boundary_result::invalid);

        auto bad = valid;
        boundary_header broken;
        std::memcpy(&broken, valid.data(), sizeof(broken));
        for (int which = 0; which < 5; ++which) {
            boundary_header modified = broken;
            if (which == 0) { modified.version++; }
            if (which == 1) { modified.magic = 0xdeadbeef; }
            if (which == 2) { modified.n_embd = -1; }
            if (which == 3) { modified.pos = -1; }
            if (which == 4) { modified.n_embd = std::numeric_limits<int32_t>::max(); }
            std::memcpy(bad.data(), &modified, sizeof(modified));
            assert(read(bad) == boundary_result::invalid);
        }

        // Old headerless Eagle3 records must not be mistaken for either current implementation.
        std::vector<uint8_t> legacy(sizeof(llama_pos) + 4 * sizeof(float), 0);
        assert(read(legacy) == boundary_result::invalid);

        auto foreign = make_blob(magic == magic_mtp ? magic_eagle3 : magic_mtp, 8);
        assert(read(foreign) == boundary_result::foreign);
        foreign.pop_back();
        assert(read(foreign) == boundary_result::invalid);

        broken.pos = std::numeric_limits<llama_pos>::max();
        std::memcpy(bad.data(), &broken, sizeof(broken));
        assert(read(bad, 0) == boundary_result::invalid);
        assert(read(bad, std::numeric_limits<llama_pos>::max()) == boundary_result::invalid);
    }
    puts("speculative boundary framing: PASS (MTP + Eagle3)");
}
