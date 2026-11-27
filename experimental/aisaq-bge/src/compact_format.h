#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace compact {
constexpr unsigned page = 4096, dimensions = 1024, degree_limit = 32;
constexpr uint64_t id_limit = uint64_t{1} << 40;
template <typename T>
T load(const void* p) {
    T value;
    std::memcpy(&value, p, sizeof value);
    return value;
}
template <typename T>
void store(void* p, T value) {
    std::memcpy(p, &value, sizeof value);
}
inline uint64_t id40(const uint8_t* p) { return load<uint32_t>(p) | (uint64_t(p[4]) << 32); }
inline void put_id40(uint8_t* p, uint64_t id) {
    if (id >= id_limit) throw std::out_of_range("node ID exceeds 40 bits");
    store(p, uint32_t(id));
    p[4] = uint8_t(id >> 32);
}
struct Geometry {
    unsigned width = 0;
    uint64_t rows = 0, first = 0, entry = 0;
    bool legacy = false;
    bool contains(uint64_t id) const { return id >= first && id - first < rows; }
    void validate() const {
        if ((width != 48 && width != 56 && width != 64) || !rows || first >= id_limit ||
            rows > id_limit - first || !contains(entry))
            throw std::runtime_error("invalid compact geometry");
    }
    uint64_t offset(uint64_t id) const {
        if (!contains(id)) throw std::out_of_range("node outside graph ID range");
        return (1 + id - first) * page;
    }
    unsigned vector_bytes() const { return width == 64 ? 1024 : 2048; }
    void encode(uint8_t* p) const {
        validate();
        std::memcpy(p, "BGELAY02", 8);
        store(p + 8, uint32_t(width));
        store(p + 16, rows);
        store(p + 24, entry);
        store(p + 32, first);
    }
    static Geometry decode(const uint8_t* p, uint64_t size) {
        Geometry g;
        g.width = load<uint32_t>(p + 8);
        if (!std::memcmp(p, "BGELAY01", 8)) {
            g.legacy = true;
            g.rows = load<uint32_t>(p + 12);
            g.entry = load<uint32_t>(p + 16);
        } else if (!std::memcmp(p, "BGELAY02", 8)) {
            g.rows = load<uint64_t>(p + 16);
            g.entry = load<uint64_t>(p + 24);
            g.first = load<uint64_t>(p + 32);
        } else
            throw std::runtime_error("unknown compact graph version");
        g.validate();
        if (size != (1 + g.rows) * page) throw std::runtime_error("truncated or oversized compact graph");
        return g;
    }
    unsigned check_record(const uint8_t* p, uint64_t id) const {
        const bool matches = legacy ? (load<uint32_t>(p) == id && p[4] == 0x35) : id40(p) == id;
        if (!matches || p[5] != 0x4b || p[6] > degree_limit || p[7] != width)
            throw std::runtime_error("malformed compact node");
        return p[6];
    }
};
}  // namespace compact
