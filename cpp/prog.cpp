#include <cstdint>
#include <future>
#include <stdexcept>
#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <tuple>
#include <functional>
#include <cmath>
#include <random>
#include <thread>
#include <array>
#include <vector>
#include <map>

#define ASSERT_MSG(cond, msg) { if (!(cond)) throw std::runtime_error("Assetion (" #cond ") failed at '" __FILE__ "':" + std::to_string(__LINE__) + "! Msg: '" + std::string(msg) + "'."); }
#define ASSERT(cond) ASSERT_MSG(cond, "")

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

double Time() {
    static auto const gtb = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - gtb).count();
}

class Timing {
public:
    Timing(std::string const & name)
        : name_(name), tb_(Time()) {}
    ~Timing() {
        std::cout << "'" << name_ << "' time " << std::fixed
            << std::setprecision(3) << (Time() - tb_) << " sec" << std::endl;
    }
private:
    std::string name_;
    double tb_ = 0;
};

class BitVector {
public:
    bool Get(u64 i) const {
        return (bytes_[i / 8] >> (i % 8)) & u8(1);
    }
    void Set(u64 i) {
        bytes_[i / 8] |= u8(1) << (i % 8);
    }
    void ReSet(u64 i) {
        bytes_[i / 8] &= ~(u8(1) << (i % 8));
    }
    void Resize(u64 size) {
        size_ = size;
        bytes_.resize((size_ + 7) / 8);
    }
    u64 Size() const { return size_; }
private:
    u64 size_ = 0;
    std::vector<u8> bytes_;
};

template <typename T>
void GenPrimes(T glast0, std::vector<T> & res) {
    using DT = u64;
    static_assert(std::is_unsigned_v<T>);
    
    res = std::vector<T>();
    
    DT const glast = glast0;
    std::vector<DT> const primo_ps = {2, 3, 5, 7};
    DT primo = 1;
    for (auto e: primo_ps)
        primo *= e;
    std::vector<DT> primo_rems;
    for (DT i = 0; i < primo; ++i) {
        bool found = false;
        for (auto e: primo_ps)
            if (i % e == 0) {
                found = true;
                break;
            }
        if (!found)
            primo_rems.push_back(i);
    }
    
    std::vector<DT> ps_first = {2};
    for (DT p = 3; p < primo; p += 2) {
        bool is_prime = true;
        for (auto d: ps_first) {
            if (d * d > p)
                break;
            if (p % d == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime)
            ps_first.push_back(p);
    }
    
    ASSERT_MSG(glast >= primo, "glast " + std::to_string(glast) + " primo " + std::to_string(primo));
    
    std::vector<DT> ps;
    for (auto p: ps_first) {
        bool is_primo = false;
        for (auto e: primo_ps)
            if (p == e) {
                is_primo = true;
                break;
            }
        if (!is_primo)
            ps.push_back(p);
    }
    
    DT const pbegin = 0 / primo * primo, pend = (glast + primo) / primo * primo;
    std::vector<BitVector> filts(primo_rems.size());
    u64 const nprimo = (pend - pbegin) / primo, begin_primo = pbegin / primo;
    for (auto & e: filts)
        e.Resize(nprimo);
    
    auto Filter = [&](size_t irem) {
        DT const rem = primo_rems.at(irem);
        auto & filt = filts.at(irem);
        DT const lim = primo * filt.Size();
        
        for (auto const p: ps) {
            u64 ibegin = 0;
            DT begin = 0;
            for (begin = p * p, ibegin = 0; begin < lim; begin += p, ++ibegin)
                if (begin % primo == rem)
                    break;
            ASSERT_MSG(ibegin < primo, "primo " + std::to_string(primo) + ", rem " + std::to_string(rem) +
                ", ibegin " + std::to_string(ibegin) + ", begin % primo = " + std::to_string(begin % primo));
            if (begin >= lim)
                continue;
            for (DT i = begin / primo; i < filt.Size(); i += p)
                filt.Set(i);
        }
    };
    
    while (!ps.empty()) {
        std::cout << "ps " << ps.size() << " " << ps.back() << std::endl;
        
        {
            std::vector<std::future<void>> asyncs;
            for (size_t irem = 0; irem < primo_rems.size(); ++irem)
                asyncs.emplace_back(std::async(std::launch::async, [&, irem]{ Filter(irem); }));
            for (auto & e: asyncs)
                e.get();
        }
        
        auto const last_p = ps.back();
        ps.clear();
        
        DT const last_end = primo * filts.at(0).Size();
        
        if (last_p * last_p >= last_end)
            break;
        
        DT const
            begin_i = (last_p + 1) / primo,
            end_i = std::min<DT>(filts.at(0).Size(), (last_p * last_p) / primo + 1);
        
        for (DT i = begin_i; i < end_i; ++i) {
            DT const i_primo = i * primo;
            bool const first = (i <= begin_i), last = (i + 1 >= end_i);
            for (size_t irem = 0; irem < filts.size(); ++irem) {
                if (filts[irem].Get(i))
                    continue;
                auto const p = i_primo + primo_rems[irem];
                if (first && p <= last_p)
                    continue;
                if (last && p > last_p * last_p)
                    break;
                ps.push_back(p);
            }
            if (!ps.empty() && ps.back() * ps.back() >= last_end)
                break;
        }
        
        ASSERT(!ps.empty());
    }
    
    ps = std::vector<DT>();
    for (auto const p: ps_first)
        res.push_back(T(p));
    for (DT i = 1; i < filts[0].Size(); ++i) {
        DT const i_primo = i * primo;
        for (size_t irem = 0; irem < filts.size(); ++irem) {
            if (filts[irem].Get(i))
                continue;
            DT const p = i_primo + primo_rems[irem];
            if (p > glast)
                continue;
            res.push_back(T(p));
        }
    }
    res.shrink_to_fit();
}

template <typename T>
void CreateLoadPrimes(size_t const bits, std::vector<T> & ps) {
    ps = std::vector<T>();
    ASSERT(bits >= 8);
    ASSERT(bits <= sizeof(T) * 8);
    static_assert(std::is_unsigned_v<T>);
    std::string fname = "primes." + std::string(bits < 10 ? " " : "") + std::to_string(bits);
    if (!std::filesystem::exists(fname)) {
        std::vector<T> primes;
        GenPrimes(T(-1) >> (sizeof(T) * 8 - bits), primes);
        ASSERT(primes.size() >= 2 && primes.at(0) == 2 && primes.at(1) == 3);
        std::vector<u8> buf(primes.size() - 2);
        for (size_t i = 2; i < primes.size(); ++i)
            buf[i - 2] = u8((primes[i] - primes[i - 1]) >> 1);
        std::ofstream f(fname, std::ios::binary);
        f.write((char*)buf.data(), buf.size());
    }
    
    u64 const fsize = std::filesystem::file_size(fname);
    std::vector<u8> buf(fsize);
    {
        std::ifstream f(fname, std::ios::binary);
        f.read((char*)buf.data(), buf.size());
    }
    ps.resize(buf.size() + 2);
    ps[0] = 2; ps[1] = 3;
    for (size_t i = 2; i < buf.size() + 2; ++i)
        ps[i] = ps[i - 1] + (T(buf[i - 2]) << 1);
    
    u64 sum = 0;
    for (auto e: ps)
        sum += e;
    
    // https://oeis.org/A130739/b130739.txt
    std::vector<std::tuple<size_t, u64>> const primes_sum = {
        {0, 0ULL}, {1, 0ULL}, {2, 5ULL}, {3, 17ULL}, {4, 41ULL}, {5, 160ULL}, {6, 501ULL}, {7, 1'720ULL}, {8, 6'081ULL},
        {9, 22'548ULL}, {10, 80'189ULL}, {11, 289'176ULL}, {12, 1'070'091ULL}, {13, 3'908'641ULL}, {14, 14'584'641ULL},
        {15, 54'056'763ULL}, {16, 202'288'087ULL}, {17, 761'593'692ULL}, {18, 2'867'816'043ULL}, {19, 10'862'883'985ULL},
        {20, 41'162'256'126ULL}, {21, 156'592'635'694ULL}, {22, 596'946'687'124ULL}, {23, 2'280'311'678'414ULL}, {24, 8'729'068'693'022ULL},
        {25, 33'483'086'021'512ULL}, {26, 128'615'914'639'624ULL}, {27, 494'848'669'845'962ULL},
        {28, 1'906'816'620'981'654ULL}, {29, 7'357'074'544'482'779ULL}, {30, 28'422'918'403'819'825ULL}, {31, 109'930'816'131'860'852ULL},
        {32, 425'649'736'193'687'430ULL}, {33, 1'649'816'561'794'735'645ULL}, {34, 6'400'753'258'957'522'036ULL},
    };
    
    auto const [ref_bits, ref_sum] = primes_sum.at(bits);
    ASSERT(ref_bits == bits);
    ASSERT_MSG(sum == ref_sum, "sum " + std::to_string(sum) + ", ref_sum " + std::to_string(ref_sum));
}

void FactorRange(u64 const limit, std::vector<u32> & fs) {
    fs = std::vector<u32>();
    fs.resize((limit + 1) / 2);
    std::vector<u32> ps;
    {
        auto const p_bits = size_t(std::log2(std::max<double>(1, std::sqrt(fs.size()))) + 1);
        Timing tim("Gen " + std::to_string(p_bits) + "-bit primes");
        CreateLoadPrimes(16, ps);
    }
    Timing tim("Factor range");
    if (!fs.empty())
        fs.at(0) = 1;
    for (auto const p: ps) {
        if (p == 2)
            continue;
        for (u64 i = p; i < fs.size() * 2; i += p * 2) {
            auto const i0 = i >> 1;
            if (!fs[i0])
                fs[i0] = p;
        }
    }
    for (auto & e: fs)
        if (!e)
            e = 1;
}

void FindSquares(u64 const N0, bool should_square, std::vector<u32> const & fs, std::vector<std::tuple<u64, u64>> & sqrs) {
    thread_local std::vector<std::tuple<u64, u8>> fc0;
    auto & fc = fc0;
    fc.clear();
    
    if (should_square)
        ASSERT(N0 <= u32(-1));
    u64 const N = should_square ? N0 * N0 : N0;
    
    auto FindCnts = [&](u64 x){
        ASSERT_MSG(x > 0 && (x >> 1) < fs.size(),
            "x " + std::to_string(x) + " fs.size() " + std::to_string(fs.size()));
        while (x > 1) {
            u32 p = 0;
            if ((x & 1) == 0)
                p = 2;
            else {
                auto const fsx = fs[x >> 1];
                p = fsx == 1 ? x : fsx;
                ASSERT(p > 1 && x % p == 0);
            }
            bool found = false;
            for (auto & [k, v]: fc)
                if (k == p) {
                    found = true;
                    ++v;
                    break;
                }
            if (!found)
                fc.push_back(std::make_tuple(p, 1));
            x /= p;
        }
    };
    
    FindCnts(N0);
    
    if (should_square)
        for (auto & [k, v]: fc)
            v *= 2;
            
    sqrs.clear();
    
    std::function<void(size_t, u64)> Iter = [&](size_t i, u64 A){
        if (i >= fc.size()) {
            u64 const B = N / A;
            ASSERT(A * B == N);
            if (A < B)
                return;
            auto d = A - B;
            if (d & 1)
                return;
            u64 const X = (A + B) / 2, Y = (A - B) / 2;
            ASSERT(N + Y * Y == X * X);
            sqrs.push_back(std::make_tuple(Y, X));
            return;
        }
        auto const [f, c] = fc[i];
        for (size_t j = 0; j <= c; ++j) {
            Iter(i + 1, A);
            A *= f;
        }
    };
    
    Iter(0, 1);
}

inline u64 ISqrt(u64 S) {
    if (S <= 1)
        return S;
    auto AbsDiff = [](auto x, auto y) {
        return x >= y ? x - y : y - x;
    };
    u64 x = S / 2, xn = 0;
    for (size_t icycle = 0; icycle < 100; ++icycle) {
        ASSERT_MSG(x > 0, "S " + std::to_string(S));
        xn = (x + S / x) / 2;
        if (AbsDiff(x, xn) <= 1) {
            u64 const y_start = std::max<u64>(xn, 2) - 2;
            ASSERT_MSG(y_start * y_start <= S, "S " + std::to_string(S));
            for (u64 y = y_start; y <= xn + 2; ++y)
                if (y * y > S)
                    return y - 1;
            ASSERT_MSG(false, "S " + std::to_string(S));
        }
        x = xn;
    }
    ASSERT_MSG(false, "S " + std::to_string(S));
}

BitVector SqrFilter() {
    Timing tim("Square filter compute");
    u64 const K = 2ULL * 2 * 3 * 3 * 5 * 7 * 11 * 13 * 17 * 19;
    BitVector bv;
    bv.Resize(K);
    for (u64 i = 0; i < K; ++i)
        bv.Set((i * i) % K);
    return bv;
}

inline std::tuple<bool, u64> IsSquare(u64 x) {
    static auto const filt = SqrFilter();
    if (!filt.Get(x % filt.Size()))
        return std::make_tuple(false, 0ULL);
    
    if (x < (1ULL << 44)) {
        auto const root = u64(std::sqrt(double(x)) + 0.5);
        return std::make_tuple(root * root == x, root);
    } else {
        auto const root = ISqrt(x);
        return std::make_tuple(root * root == x, root);
    }
}

void FindSquaresSlow(u64 const N, std::vector<std::tuple<u64, u64>> & sqrs) {
    for (u64 Y = 0; Y < N; ++Y) {
        auto const [is_sqr, root] = IsSquare(N + Y * Y);
        if (!is_sqr)
            continue;
        sqrs.push_back(std::make_tuple(Y, root));
    }
}

void Solve(u64 limit, size_t const L = 4) {
    Timing gtim("Total Solve");
    
    ASSERT(limit <= u64(u32(-1)) + 1);
    
    size_t constexpr max_L = 4;
    ASSERT(L <= max_L);
    
    std::vector<u32> fs;
    FactorRange(limit, fs);
    
    auto FName = [&](size_t l) {
        return "cpp_solutions." + std::to_string(l) + "." + std::to_string(limit);
    };
    
    struct __attribute__((packed)) Entry {
        u32 x = 0;
    };
    
    std::vector<std::array<Entry, max_L>> A;
    
    size_t start_il = 0;
    
    if (0 && std::filesystem::exists(FName(3))) {
        start_il = 3;
    } else {
        for (u64 i = 1; i < limit; ++i) {
            A.resize(A.size() + 1);
            A.back()[0].x = i;
        }
        start_il = 1;
    }
    
    size_t constexpr nblocks = 1 << 9;
    size_t const cpu_count = std::thread::hardware_concurrency();
    
    for (size_t il = start_il; il < L; ++il) {
        u64 const block = (A.size() + nblocks - 1) / nblocks;
        std::vector<std::future<std::tuple<std::pair<u64, u64>, std::vector<std::array<Entry, max_L>>>>> asyncs;
        
        std::map<u64, std::tuple<std::pair<u64, u64>, std::vector<std::array<Entry, max_L>>>> Ats;
        
        double const tb = Time();
        std::atomic<double> avg_sqrs = 0, avg_sqrs_cnt = 0;
        u64 num_new_sols = 0;
        
        for (u64 iblock = 0, iblock_iter = 0; iblock < A.size(); iblock += block, ++iblock_iter) {
            u64 const cur_size = std::min<u64>(A.size() - iblock, block);
            
            asyncs.push_back(std::async(std::launch::async, [&, il, iblock, cur_size]{
                std::vector<std::array<Entry, max_L>> At;
                if (cur_size == 0)
                    return std::make_tuple(std::pair{iblock, iblock + cur_size}, std::move(At));
                thread_local std::vector<std::tuple<u64, u64>> sqrs0;
                auto & sqrs = sqrs0;
                double avg_sqrs0 = 0, avg_sqrs_cnt0 = 0;
                for (u64 i = iblock; i < iblock + cur_size; ++i) {
                    sqrs.clear();
                    FindSquares(A[i][il - 1].x, true, fs, sqrs);
                    std::sort(sqrs.begin(), sqrs.end());
                    avg_sqrs0 += sqrs.size();
                    avg_sqrs_cnt0 += 1;
                    u64 const px2 = u64(A[i][il - 1].x) * A[i][il - 1].x;
                    for (auto const [y, x]: sqrs) {
                        if (y == 0)
                            continue;
                        if (x >= limit)
                            continue;
                        u64 const x2 = u64(x) * x;
                        bool bad = false;
                        for (size_t jl = 0; jl + 1 < il; ++jl)
                            if (!std::get<0>(IsSquare(x2 - u64(A[i][jl].x) * A[i][jl].x))) {
                                bad = true;
                                break;
                            }
                        if (bad)
                            continue;
                        At.resize(At.size() + 1);
                        std::memcpy(&At.back()[0], &A[i][0], int(((u8*)&A[i][il]) - ((u8*)&A[i][0])));
                        ASSERT_MSG(x <= u64(u32(-1)), "x_prev " + std::to_string(A[i][il - 1].x) +
                            ", y " + std::to_string(y) + ", x " + std::to_string(x));
                        At.back()[il].x = x;
                    }
                }
                avg_sqrs += avg_sqrs0;
                avg_sqrs_cnt += avg_sqrs_cnt0;
                return std::make_tuple(std::pair{iblock, iblock + cur_size}, std::vector<std::array<Entry, max_L>>(At));
            }));
            
            while (asyncs.size() >= cpu_count * 2 ||
                    iblock + block >= A.size() && asyncs.size() > 0) {
                for (ptrdiff_t ie = ptrdiff_t(asyncs.size()) - 1; ie >= 0; --ie) {
                    auto & e = asyncs[ie];
                    if (e.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready)
                        continue;
                    auto const r = e.get();
                    asyncs.erase(asyncs.begin() + ie);
                    num_new_sols += std::get<1>(r).size();
                    Ats[std::get<0>(r).first] = std::move(r);
                }
                std::this_thread::yield();
            }
            
            if (((iblock_iter + 1) & ((1 << 2) - 1)) == 0 || iblock + block >= A.size()) {
                double const ratio_passed = std::max<double>(1, double(iblock + cur_size) - double(asyncs.size() * block)) / A.size(),
                    ratio_left = 1.0 - ratio_passed;
                std::cout << "L " << (il + 1) << "/" << L << ", i " << std::setfill(' ') << std::setw(6)
                    << (iblock + cur_size) / 1'000'000 << "/" << std::fixed << std::setprecision(2) << (A.size() / 1'000'000.0)
                    << " M, new " << std::fixed << std::setprecision(2) << (num_new_sols / 1'000'000.0) << " M, avg sqrs "
                    << std::fixed << std::setprecision(2) << (avg_sqrs / std::max<double>(1, avg_sqrs_cnt))
                    << std::setprecision(1) << ", ELA " << Time() / 60.0 << " mins, ETA "
                    << std::setprecision(1) << (ratio_left / ratio_passed * (Time() - tb) / 60.0) << " mins" << std::endl;
            }
        }
        
        ASSERT(asyncs.empty());
        
        A = std::vector<std::array<Entry, max_L>>();
        
        std::vector<std::array<Entry, max_L>> A2;
        u64 next = 0;
        for (auto const [k, v]: Ats) {
            ASSERT(k == next);
            auto const & [a, b] = v;
            ASSERT(k == a.first);
            A2.insert(A2.end(), b.begin(), b.end());
            next = a.second;
        }
        Ats.clear();
        A = std::move(A2);
        std::cout << std::endl << "L " << (il + 1) << " has " << std::fixed << A.size() / 1'000'000 << "."
            << std::setfill('0') << std::setw(6) << (A.size() % 1'000'000) << " M solutions" << std::endl << std::endl;
        
        if (il >= 2) {
            std::ofstream f(FName(il + 1));
            for (auto e: A) {
                size_t const jend = il + 1;
                for (size_t j = 0; j < jend; ++j)
                    f << e[j].x << (j + 1 >= jend ? "" : ", ");
                f << std::endl;
            }
            std::cout << "Solutions for L=" << (il + 1) << " saved to '" << FName(il + 1) << "'" << std::endl;
        }
    }
}

void Test() {
    u64 const limit = 1ULL << 11;
    
    std::vector<u32> fs;
    FactorRange(limit, fs);
    double sqrs_sum = 0, sqrs_cnt = 0;
    std::vector<std::tuple<u64, u64>> sqrs0, sqrs1;
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<u64> distr(1, limit - 1);
    size_t const iend = 1 << 8;
    int report_i = -int(iend);
    Timing tim("Test");
    for (int i = 0; i < iend; ++i) {
        if (i - report_i >= iend / 10) {
            std::cout << "test " << i << ", " << std::flush;
            report_i = i;
        }
        auto const N = distr(rng);
        sqrs0.clear();
        FindSquares(N, true, fs, sqrs0);
        std::sort(sqrs0.begin(), sqrs0.end());
        sqrs1.clear();
        FindSquaresSlow(N * N, sqrs1);
        auto ToStr = [&](auto const & s) {
            std::string r;
            for (auto e: s) {
                auto const [k, v] = e;
                r += std::to_string(k) + "/" + std::to_string(v) + ", ";
            }
            return r;
        };
        ASSERT_MSG(sqrs0 == sqrs1, "N " + std::to_string(N) +
            ", sqrs0 " + ToStr(sqrs0) + ", sqrs1 " + ToStr(sqrs1));
    }
    return;
    
    #if 0
    double tb = Time();
    for (size_t N = 1; N < fs.size(); ++N) {
        if (N % 5'000'000 == 0 || N + 1 >= fs.size()) {
            std::cout << N / 1'000'000 << " M (avg " << std::fixed << std::setprecision(2)
                << sqrs_sum / std::max<double>(1, sqrs_cnt) << ", time "
                << (Time() - tb) << " sec), " << std::flush;
        }
        sqrs.clear();
        FindSquares(N, true, fs, sqrs);
        sqrs_sum += sqrs.size();
        sqrs_cnt += 1;
    }
    #endif
}

int main() {
    try {
        //Test(); return 0;
        Solve((1ULL << 32) - 2);
    } catch (std::exception const & ex) {
        std::cout << "Exception: " << ex.what() << std::endl;
    }
}