#pragma once
#include <sys/resource.h>
#include <time.h>

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

static void capture_window() {
    const char* prefix = std::getenv("ANN_WINDOW");
    const char* device = std::getenv("ANN_BLOCK_DEVICE");
    std::string name = device ? device : "nvme0n1";
    if (name.empty() || name.find('/') != std::string::npos) {
        throw std::runtime_error("ANN_BLOCK_DEVICE must name a block device");
    }
    std::ifstream state("/sys/class/block/" + name + "/stat");
    std::string line;
    if (!std::getline(state, line)) {
        throw std::runtime_error("Could not read block device statistics for " + name);
    }
    timespec stamp{};
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    std::ofstream out(prefix, std::ios::app);
    out << stamp.tv_sec * 1000000000ULL + stamp.tv_nsec << ' ' << line << '\n';
    std::ofstream cpu(std::string(prefix) + ".cpu", std::ios::app);
    cpu << usage.ru_utime.tv_sec * 1000000LL + usage.ru_utime.tv_usec << ' '
        << usage.ru_stime.tv_sec * 1000000LL + usage.ru_stime.tv_usec << '\n';
    if (!out || !cpu) throw std::runtime_error("Could not write counter window");
}
