#include <cstdint>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <array>
#include <unordered_map>
#include <tuple>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <cmath>

int main(int argc, char ** argv) {
    using u8 = uint8_t;
    using i64 = int64_t;
    using u64 = uint64_t;
    
    std::string fname(argc >= 2 ? argv[1] : "input.txt");
    
    std::ifstream f(fname);
    if (!f.is_open()) {
        std::cout << "Failed to open file '" << fname << "'." << std::endl;
        return -1;
    }
    auto const gtb = std::chrono::high_resolution_clock::now();
    auto Time = [gtb]() -> double {
        return std::llround(std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - gtb).count() * 1000.0) / 1000.0;
    };
    std::vector<std::array<u64, 3>> v;
    double tb = 0;
    {
        u64 const file_size = std::filesystem::file_size(fname);
        std::string text(file_size, ' ');
        f.read((char*)text.data(), text.size());
        u64 prev = 0;
        tb = Time();
        for (size_t icycle = 0;; ++icycle) {
            if ((icycle & ((1ULL << 24) - 1)) == 0)
                std::cout << "read " << (icycle >> 20) << " M " << (Time() - tb) << " sec, " << std::flush;
            u64 next = text.find('\n', prev);
            if (next == std::string::npos)
                next = file_size;
            u64 const
                first_comma = text.find(',', prev),
                second_comma = text.find(',', first_comma + 1);
            v.push_back({});
            std::array<u64, 3> poss = {prev, first_comma + 1, second_comma + 1};
            for (size_t i = 0; i < 3; ++i) {
                char * pend = nullptr;
                auto const val = std::strtoll(text.c_str() + poss[i], &pend, 10);
                if (val == 0) {
                    v.pop_back();
                    break;
                }
                v.back()[i] = val;
            }
            if (next >= file_size)
                break;
            prev = next + 1;
        }
    }
    /*
    for (size_t i = 0;; ++i) {
        if (i % 100'000 == 0)
            std::cout << "Read Line " << i / 1'000 << " K, " << std::flush;
        std::string line;
        std::getline(f, line);
        std::stringstream ss;
        ss.str(line);
        std::array<u64, 3> a{};
        char comma = 0;
        ss >> a[0] >> comma >> a[1] >> comma >> a[2];
        if (!f)
            break;
        if (a[2] == 0)
            continue;
        v.push_back(a);
    }
    std::cout << std::endl;
    */
    std::sort(v.begin(), v.end(),
        [](auto const & x, auto const & y) -> bool {
            return x < y;
        });
    
    struct Hasher {
        static u64 FnvHash(void const * data, size_t size, u64 prev = u64(-1)) {
            // http://www.isthe.com/chongo/tech/comp/fnv/#FNV-param
            u64 constexpr
                fnv_prime = 1099511628211ULL,
                fnv_offset_basis = 14695981039346656037ULL;
            
            u64 hash = prev == u64(-1) ? fnv_offset_basis : prev;
            
            for (size_t i = 0; i < size; ++i) {
                hash ^= ((u8*)data)[i];
                hash *= fnv_prime;
            }
            
            return hash;
        }
        
        size_t operator () (std::tuple<u64, u64> const & x) const {
            return FnvHash(&x, sizeof(x));
            //auto const h0 = h_(std::get<0>(x)); return ((h0 << 13) | (h0 >> (sizeof(h0) * 8 - 13))) + h_(std::get<1>(x));
        }
    };
    std::unordered_map<std::tuple<u64, u64>, std::tuple<size_t, size_t>, Hasher> m;
    std::tuple<u64, u64> prev = std::make_tuple(v.at(0)[0], v.at(0)[1]);
    size_t start = 0;
    tb = Time();
    for (size_t i = 0; i < v.size(); ++i) {
        if ((i & ((1ULL << 24) - 1)) == 0)
            std::cout << "map " << (i >> 20) << " M " << (Time() - tb) << " sec, " << std::flush;
        auto const next = std::make_tuple(v[i][0], v[i][1]);
        if (prev == next)
            continue;
        m[prev] = std::make_tuple(start, i);
        prev = next;
        start = i;
    }
    m[prev] = std::make_tuple(start, v.size());
    std::ofstream fout(argc >= 3 ? argv[2] : "output.txt");
    size_t icycle = 0;
    tb = Time();
    for (auto const & a: v) {
        if ((icycle & ((1ULL << 24) - 1)) == 0)
            std::cout << "find " << (icycle >> 20) << " M " << (Time() - tb) << " sec, " << std::flush;
        ++icycle;
        auto const it = m.find(std::make_tuple(a[1], a[2]));
        if (it == m.end())
            continue;
        for (size_t i = std::get<0>(it->second); i < std::get<1>(it->second); ++i)
            fout << a[0] << ", " << a[1] << ", " << a[2] << ", " << v[i][2] << std::endl;
    }
    return 0;
}