#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace knowhere {
struct KnowhereConfig {
    static void SetBuildThreadPoolSize(size_t);
};
}  // namespace knowhere

int generate_pq_pivots(const float*, size_t, unsigned, unsigned, unsigned, unsigned, std::string, bool);
template <typename T>
int generate_pq_data_from_pivots(std::string, unsigned, unsigned, std::string, std::string);
template <typename T>
void gen_random_slice(std::string, double, std::unique_ptr<float[]>&, size_t&, size_t&,
                      std::vector<uint32_t>*);

int main(int argc, char** argv) {
    try {
        if (argc < 4)
            throw std::runtime_error(
                "pq sample BASE.fbin IDS.u32 | pq train BASE.fbin PREFIX {48|56} IDS.u32 | pq encode "
                "BASE.fbin PREFIX {48|56}");
        const std::string command = argv[1], base = argv[2], dest = argv[3];
        std::ifstream file(base, std::ios::binary);
        uint32_t shape[2];
        if (!file.read(reinterpret_cast<char*>(shape), sizeof(shape)) || !shape[0] || shape[1] != 1024) {
            throw std::runtime_error("Expected a nonempty 1024-dimensional float32 base");
        }
        knowhere::KnowhereConfig::SetBuildThreadPoolSize(64);
        if (command == "sample" && argc == 4) {
            std::unique_ptr<float[]> sample;
            size_t n = 0, dimension = 0;
            std::vector<uint32_t> ids;
            double fraction = std::min(1.0, 256000.0 / shape[0]);
            gen_random_slice<float>(base, fraction, sample, n, dimension, &ids);
            if (ids.size() != n || n < 256 || dimension != shape[1])
                throw std::runtime_error("Invalid training sample");
            std::ofstream result(dest, std::ios::binary);
            result.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(uint32_t));
            result.close();
            if (!result) throw std::runtime_error("Writing training IDs failed");
            return 0;
        }
        if (argc < 5) throw std::runtime_error("Missing PQ width");
        unsigned width = std::stoul(argv[4]);
        if (width != 48 && width != 56) throw std::runtime_error("PQ width must be 48 or 56");
        auto pivots = dest + "_pq_pivots.bin";
        if (command == "encode" && argc == 5)
            return generate_pq_data_from_pivots<float>(base, 256, width, pivots,
                                                       dest + "_pq_compressed.bin");
        if (command != "train" || argc != 6)
            throw std::runtime_error("Invalid PQ command or missing sample file");
        std::ifstream input(argv[5], std::ios::binary | std::ios::ate);
        auto length = input.tellg();
        if (length < 256 * 4 || length % 4 || length / 4 > shape[0])
            throw std::runtime_error("Invalid training IDs file length");
        std::vector<uint32_t> ids(static_cast<size_t>(length) / 4);
        input.seekg(0);
        if (!input.read(reinterpret_cast<char*>(ids.data()), length))
            throw std::runtime_error("Reading training IDs failed");
        std::sort(ids.begin(), ids.end());
        if (ids.back() >= shape[0] || std::adjacent_find(ids.begin(), ids.end()) != ids.end())
            throw std::runtime_error("Invalid or duplicate training ID");
        std::vector<float> vectors(ids.size() * 1024);
        for (size_t i = 0; i < ids.size(); ++i) {
            file.seekg(8ULL + uint64_t(ids[i]) * 4096);
            if (!file.read(reinterpret_cast<char*>(vectors.data() + i * 1024), 4096))
                throw std::runtime_error("Reading training vector failed");
        }
        return generate_pq_pivots(vectors.data(), ids.size(), 1024, 256, width, 12, pivots, true);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
