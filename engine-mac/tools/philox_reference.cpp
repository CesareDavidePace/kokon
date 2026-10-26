// sd.cpp's default noise generator (PhiloxRNG, src/core/rng_philox.hpp of the fork's `kokon` branch), compiled
// as Kokon's macOS engine is (Apple clang, Release, default float contraction), for `kokon-engine-mac check-noise`.
// Writes raw little-endian floats: for each seed below, a 1024² latent's draw (262 144) then the next draw (1 000).
//
//   clang++ -std=c++17 -O3 -I <stable-diffusion.cpp>/src engine-mac/tools/philox_reference.cpp -o philox
//   ./philox > philox.bin
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "core/rng_philox.hpp"
int main(int argc, char** argv) {
    uint64_t seeds[] = {0, 1, 42, 711449798ull, 4001ull, 0xFFFFFFFFull, 0x123456789ABCDEFull, 18446744073709551615ull};
    for (uint64_t s : seeds) {
        PhiloxRNG rng(s);
        auto a = rng.randn(262144);   // 1024²: 64 × 64 × 64
        auto b = rng.randn(1000);     // the next draw (offset 1)
        fwrite(a.data(), 4, a.size(), stdout);
        fwrite(b.data(), 4, b.size(), stdout);
    }
}
