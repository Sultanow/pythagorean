#include <cstdint>
#include <cstring>
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
#include <type_traits>
#include <sstream>
#include <memory>
#include <atomic>
#include <mutex>

#define LIMIT (1ULL << 34)
#define IS_128 1
#define MBLOCK (1ULL << 27)

#define SUPPORT_ZSTD 1
#define ZSTD_COMPRESSION_LEVEL 3
#define ZSTD_NUM_THREADS 0

#if SUPPORT_ZSTD
    #include <zstd.h>
#endif

#define ASSERT_MSG(cond, msg) { if (!(cond)) throw std::runtime_error("Assetion (" #cond ") failed at '" __FILE__ "':" + std::to_string(__LINE__) + "! Msg: '" + std::string(msg) + "'."); }
#define ASSERT(cond) ASSERT_MSG(cond, "")

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using i128 = signed __int128;
using u128 = unsigned __int128;

#if IS_128
    using WordT = u64;
    using DWordT = u128;
#else
    using WordT = u32;
    using DWordT = u64;
#endif

double Time() {
    static auto const gtb = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - gtb).count();
}

static std::ofstream & LogFile() {
    static std::ofstream f("cpp_solutions.log");
    return f;
}

class Log {
public:
    Log() : ss_(std::make_shared<std::stringstream>()) {}
    ~Log() {
        std::cout << ss_->str() << std::flush;
        LogFile() << ss_->str() << std::flush;
    }
    std::stringstream & Get() const { return *ss_; }
private:
    std::shared_ptr<std::stringstream> ss_;
};

#define LOG Log().Get()

class RunInDestr {
public:
    RunInDestr(std::function<void()> const & f)
        : f_(f) {}
    ~RunInDestr() { f_(); }
private:
    std::function<void()> f_;
};

class Timing {
public:
    Timing(std::string const & name)
        : name_(name), tb_(Time()) {}
    void SetName(std::string const & name) { name_ = name; }
    ~Timing() {
        LOG << "'" << name_ << "' time " << std::fixed
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
        //LOG << "ps " << ps.size() << " " << ps.back() << std::endl;
        
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
    fs = std::decay_t<decltype(fs)>();
    fs.resize((limit + 1) / 2);
    std::vector<u32> ps;
    {
        auto const p_bits = size_t(std::log2(std::max<double>(1, std::sqrt(fs.size()))) + 1);
        Timing tim("Generate " + std::to_string(p_bits) + "-bit primes");
        CreateLoadPrimes(16, ps);
    }
    Timing tim("Factor Range calc total");
    double report_time = -1000, tb = Time();
    if (!fs.empty())
        fs.at(0) = 1;
    u64 ip = 0;
    for (auto const p: ps) {
        if (p == 2)
            continue;
        for (u64 i = p; i < fs.size() * 2; i += p * 2) {
            auto & e = fs[i >> 1];
            if (e == 0)
                e = p;
        }
        if ((ip & 0xFF) == 0 && Time() - report_time >= 30) {
            LOG << "Factor range calc " << std::setfill(' ') << std::setw(std::to_string(ps.size() >> 10).size())
                << (ip >> 10) << "/" << (ps.size() >> 10) << " K, ELA "
                << std::fixed << std::setprecision(1) << (Time() - tb) / 60.0 << " mins" << std::endl;
            report_time = Time();
        }
        ++ip;
    }
    for (auto & e: fs)
        if (e == 0)
            e = 1;
}

void FindSquares(u64 const N0, bool should_square, std::vector<u32> const & fs, std::vector<std::tuple<u64, u64>> & sqrs, u64 limit = (u64(-1) >> 1)) {
    thread_local std::vector<std::tuple<u64, u16>> fc0;
    auto & fc = fc0;
    fc.clear();

    ASSERT(limit <= (u64(-1) >> 1));
    ASSERT(should_square);
    
    //if (should_square) ASSERT(N0 <= u32(-1));
    DWordT const N = should_square ? DWordT(N0) * N0 : N0;
    
    auto FindCnts = [&](u64 x){
        ASSERT_MSG(x > 0 && (x >> 1) < fs.size(),
            "x " + std::to_string(x) + " fs.size() " + std::to_string(fs.size()));
        while (x > 1) {
            u64 p = 0;
            if ((x & 1) == 0)
                p = 2;
            else {
                auto const fsx = fs[x >> 1];
                p = fsx == 1 ? x : fsx;
                ASSERT(p > 1);
                //ASSERT(x % p == 0);
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
    
    std::function<void(size_t, u64)> Iter = [&](size_t i, DWordT B){
        if (i >= fc.size()) {
            if (B > DWordT(u64(-1)))
                return;
            u64 const B64 = u64(B);
            DWordT const A = N / B64;
            ASSERT(A * B64 == N);
            if (A < B)
                return;
            ASSERT(((A - B64) & 1) == 0);
            DWordT const X = (A + B64) >> 1;
            if (X >= DWordT(limit))
                return;
            DWordT const Y = (A - B64) >> 1;
            ASSERT(N + DWordT(u64(Y)) * u64(Y) == DWordT(u64(X)) * u64(X));
            sqrs.push_back(std::make_tuple(u64(Y), u64(X)));
            return;
        }
        auto const [f, c] = fc[i];
        if (f == 2)
            B *= f;
        for (size_t j = (f == 2 ? 1 : 0); j <= (f == 2 ? c - 1 : c); ++j) {
            Iter(i + 1, B);
            B *= f;
        }
    };
    
    Iter(0, 1);
}

template <typename T>
std::string NumToStr(T x) {
    std::stringstream ss;
    auto constexpr mod = 1'000'000'000'000'000'000ULL;
    auto const hi = u64(x / mod);
    if (hi > 0)
        ss << hi;
    ss << u64(x % mod);
    return ss.str();
}

template <typename T, typename DT>
inline T ISqrt(DT const & S) {
    if (S <= 1)
        return S;
    auto AbsDiff = [](auto const & x, auto const & y) {
        return x >= y ? x - y : y - x;
    };
    if (sizeof(DT) > 8)
        ASSERT_MSG(S < (DT(-1) >> 5), "S " + NumToStr(S));
    T x = T(std::min<DT>(T(-1) >> 2, S >> 1)), xn = 0;
    for (size_t icycle = 0; icycle < 100; ++icycle) {
        ASSERT_MSG(x > 0, "S " + NumToStr(S));
        xn = (x + S / x) >> 1;
        if (AbsDiff(x, xn) <= 1) {
            u64 const y_start = std::max<u64>(xn, 2) - 2;
            ASSERT_MSG(y_start * y_start <= S, "S " + NumToStr(S));
            for (u64 y = y_start; y <= xn + 2; ++y)
                if (y * y > S)
                    return y - 1;
            ASSERT_MSG(false, "S " + NumToStr(S));
        }
        x = xn;
    }
    ASSERT_MSG(false, "S " + NumToStr(S));
}

static u64 constexpr sqrf_K = 2ULL * 2 * 3 * 3 * 5 * 7 * 11 * 13 * 17 * 19;

BitVector SqrFilter() {
    BitVector bv;
    {
        Timing tim("Square filter compute");
        bv.Resize(sqrf_K);
        for (u64 i = 0; i < sqrf_K; ++i)
            bv.Set(u64((i * i) % sqrf_K));
    }
    u64 cnt1 = 0;
    for (u64 i = 0; i < sqrf_K; ++i)
        cnt1 += u8(bv.Get(i));
    LOG << "Square filter ratio " << std::fixed
        << std::setprecision(5) << double(cnt1) / sqrf_K << std::endl;
    return bv;
}

template <typename DT>
inline std::tuple<bool, u64> IsSquare(DT const & x) {
    static auto const filt = SqrFilter();
    if (!filt.Get(u64(x % sqrf_K)))
        return std::make_tuple(false, 0ULL);
    
    if (x < (1ULL << 44)) {
        auto const root = u64(std::sqrt(double(x)) + 0.5);
        return std::make_tuple(DT(root) * root == x, root);
    } else {
        auto const root = ISqrt<u64, DT>(x);
        return std::make_tuple(DT(root) * root == x, root);
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

#if SUPPORT_ZSTD

#define ZCHECK(code) { auto res = (code); ASSERT_MSG(!ZSTD_isError(res), std::string("ZSTD error: Code ") + std::to_string(res) + ", Msg '" + ZSTD_getErrorName(res) + "'."); }

class StreamCompressor {
public:
    static size_t constexpr
        compression_level = ZSTD_COMPRESSION_LEVEL, rbuf_size0 = 1 << 25, wbuf_size0 = rbuf_size0 / 2;
    using OutF = std::function<void(u8 const * ptr, size_t size)>;
    StreamCompressor(OutF const & outf, size_t compression_level_inp = size_t(-1))
        : rbuf_size(std::max<size_t>(rbuf_size0, ZSTD_CStreamInSize())),
          wbuf_size(std::max<size_t>(wbuf_size0, ZSTD_CStreamOutSize())),
          outf_(outf) {
        ctx_ = ZSTD_createCStream();
        ASSERT(ctx_);
        if (compression_level_inp == size_t(-1))
            compression_level_inp = compression_level;
        ZCHECK(ZSTD_initCStream(ctx_, compression_level_inp));
    }
    void Write(u8 const * ptr, size_t size) {
        while (size > 0) {
            size_t const port = std::min<size_t>(rbuf_size - rbuf_.size(), size);
            rbuf_.insert(rbuf_.end(), ptr, ptr + port);
            size -= port;
            ptr += port;
            if (rbuf_.size() >= rbuf_size)
                Compress();
        }
    }
    ~StreamCompressor() noexcept(false) {
        Compress(true);
        ZCHECK(ZSTD_freeCStream(ctx_));
    }
private:
    void Compress(bool end = false) {
        if (rbuf_.size() == 0 && !end)
            return;
        ASSERT(wbuf_.size() == 0);
        wbuf_.resize(wbuf_size);
        ZSTD_inBuffer in{.src = rbuf_.data(), .size = rbuf_.size(), .pos = 0};
        ZSTD_outBuffer out{.dst = wbuf_.data(), .size = wbuf_.size(), .pos = 0};
        ZCHECK(ZSTD_compressStream2(ctx_, &out, &in, end ? ZSTD_e_end : ZSTD_e_continue));
        ASSERT(in.pos <= in.size && in.size == rbuf_.size());
        ASSERT(out.pos <= out.size && out.size == wbuf_.size());
        if (!end) {
            ASSERT(in.pos > 0);
            ASSERT(out.pos > 0);
        }
        if (out.pos > 0)
            outf_(wbuf_.data(), out.pos);
        rbuf_.erase(rbuf_.begin(), rbuf_.begin() + in.pos);
        wbuf_.clear();
        if (end)
            ASSERT(rbuf_.empty());
    }

    size_t const rbuf_size = 0, wbuf_size = 0;    
    ZSTD_CStream * ctx_ = nullptr;
    std::vector<u8> rbuf_, wbuf_;
    OutF outf_;
};

class StreamDeCompressor {
public:
    static size_t constexpr
        rbuf_size0 = 1 << 24, wbuf_size0 = rbuf_size0 * 2;
    using InpF = std::function<size_t(u8 * ptr, size_t size)>;
    StreamDeCompressor(InpF const & inpf)
        : rbuf_size(std::max<size_t>(rbuf_size0, ZSTD_DStreamInSize())),
          wbuf_size(std::max<size_t>(wbuf_size0, ZSTD_DStreamOutSize())),
          inpf_(inpf) {
        ctx_ = ZSTD_createDStream();
        ASSERT(ctx_);
        ZCHECK(ZSTD_initDStream(ctx_));
    }
    size_t Read(u8 * ptr, size_t size) {
        size_t const start_size = size;
        while (size > 0) {
            if (wbuf_.size() == 0) {
                DeCompress();
                if (wbuf_.empty())
                    break;
            }
            size_t const port = std::min<size_t>(wbuf_.size(), size);
            std::memcpy(ptr, wbuf_.data(), port);
            wbuf_.erase(wbuf_.begin(), wbuf_.begin() + port);
            size -= port;
            ptr += port;
        }
        return start_size - size;
    }
    ~StreamDeCompressor() noexcept(false) {
        ZCHECK(ZSTD_freeDStream(ctx_));
    }
private:
    void DeCompress() {
        if (wbuf_.size() >= wbuf_size)
            return;
        ASSERT(wbuf_.empty());
        wbuf_.resize(wbuf_size);
        
        if (rbuf_.size() < rbuf_size) {
            ASSERT(rbuf_.size() <= rbuf_size);
            size_t const rbuf_before = rbuf_.size();
            rbuf_.resize(rbuf_size);
            size_t const readed = inpf_(rbuf_.data() + rbuf_before, rbuf_.size() - rbuf_before);
            ASSERT(readed <= rbuf_.size() - rbuf_before);
            if (readed < rbuf_.size() - rbuf_before)
                rbuf_.resize(rbuf_before + readed);
        }
        
        ZSTD_inBuffer in{.src = rbuf_.data(), .size = rbuf_.size(), .pos = 0};
        ZSTD_outBuffer out{.dst = wbuf_.data(), .size = wbuf_.size(), .pos = 0};
        ZCHECK(ZSTD_decompressStream(ctx_, &out, &in));
        ASSERT(in.pos <= in.size && in.size == rbuf_.size());
        ASSERT(out.pos <= out.size && out.size == wbuf_.size());
        if (!rbuf_.empty()) {
            ASSERT(in.pos > 0);
            //ASSERT(out.pos > 0);
        }
        rbuf_.erase(rbuf_.begin(), rbuf_.begin() + in.pos);
        wbuf_.resize(out.pos);
        //LOG << "DeCompress in " << in.pos << " out " << out.pos << ", ";
    }
    
    size_t const rbuf_size = 0, wbuf_size = 0;    
    ZSTD_DStream * ctx_ = nullptr;
    std::vector<u8> rbuf_, wbuf_;
    InpF inpf_;
};

#endif

class FileSeqWriter {
public:
    using Filter = StreamCompressor;
    static u64 constexpr buf_size = 1ULL << 28;
    FileSeqWriter(std::string const & fname, size_t compression_level = size_t(-1), size_t num_threads = size_t(-1))
        : compression_level_(compression_level), num_threads_(num_threads), f_(fname, std::ios::binary) {
        if (num_threads_ == size_t(-1))
            num_threads_ = std::thread::hardware_concurrency();
        ASSERT(f_.is_open());
    }
    ~FileSeqWriter() { Flush(); }
    void Flush() {
        if (!buf_.empty()) {
            AddTask(buf_.data(), buf_.size());
            buf_.clear();
        }
        Cleanup(true);
    }
    template <typename T>
    void Write(T const * ptr, size_t size) {
        WriteB((u8*)ptr, size * sizeof(T));
    }
    void WriteB(u8 const * ptr, size_t size) {
        while (size > 0) {
            u64 const port = std::min<u64>(buf_size - buf_.size(), size);
            buf_.insert(buf_.end(), ptr, ptr + port);
            if (buf_.size() >= buf_size) {
                AddTask(buf_.data(), buf_.size());
                buf_.clear();
            }
            ptr += port;
            size -= port;
        }
    }
    u64 Size() const { return fsize_; }
private:
    void AddTask(u8 const * ptr, size_t size) {
        do {
            Cleanup();
        } while (filts_.size() >= num_threads_);
        auto inp  = std::make_shared<std::vector<u8>>(ptr, ptr + size);
        auto out  = std::make_shared<std::vector<u8>>();
        auto filt = std::make_shared<Filter>([out](u8 const * ptr, size_t size) {
            out->insert(out->end(), ptr, ptr + size);
        }, compression_level_);
        auto async = std::make_shared<std::future<void>>(std::async(std::launch::async,
            [inp, filt]() mutable {
                filt->Write(inp->data(), inp->size());
                filt.reset();
                *inp = std::vector<u8>();
            }));
        filts_.push_back(std::make_tuple(out, async));
    }
    void Cleanup(bool end = false) {
        while (true) {
            if (filts_.empty())
                break;
            auto & [out, async] = filts_.front();
            if (async->wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                async->get();
                f_.write((char*)out->data(), out->size());
                fsize_ += out->size();
                filts_.erase(filts_.begin());
            } else if (!end)
                break;
            std::this_thread::yield();
        }
    }

    u64 compression_level_ = 0, num_threads_ = 0, fsize_ = 0;
    std::ofstream f_;
    std::vector<u8> buf_;
    std::vector<std::tuple<std::shared_ptr<std::vector<u8>>, std::shared_ptr<std::future<void>>>> filts_;
};

class FileSeqReader {
public:
    using Filter = StreamDeCompressor;
    FileSeqReader(std::string const & fname)
        : fname_(fname), fsize_(std::filesystem::file_size(fname_)),
          f_(fname_, std::ios::binary), filt_(
            [this](u8 * ptr, size_t size){
                return this->FRead(ptr, size); }) {
        ASSERT(f_.is_open());
    }
    template <typename T>
    size_t Read(T * ptr, size_t size) {
        size_t const readed = filt_.Read((u8*)ptr, size * sizeof(T));
        ASSERT(readed % sizeof(T) == 0);
        return readed / sizeof(T);
    }
    u64 Size() const { return fsize_; }
    u64 CReaded() const { return crsize_; }
private:
    size_t FRead(u8 * ptr, size_t size) {
        f_.clear();
        f_.read((char*)ptr, size);
        i64 const readed = f_.gcount();
        ASSERT(readed >= 0);
        crsize_ += readed;
        return size_t(readed);
    }
    std::string fname_;
    u64 fsize_ = 0, crsize_ = 0;
    std::ifstream f_;
    Filter filt_;
};

void FactorRangeCreateLoad(u64 const limit, std::vector<u32> & fs) {
    size_t constexpr compr_level = 5;
    
    fs = std::decay_t<decltype(fs)>();
    std::string fname = "factor_range." + std::to_string(limit) + ".zst";
    if (!std::filesystem::exists(fname)) {
        std::vector<u32> fs0;
        FactorRange(limit, fs0);
        FileSeqWriter fw(fname, compr_level);
        Timing tim("Factor Range write total (<ERROR> compressed-MiB)");
        RunInDestr tim_set_name([&]{
            tim.SetName("Factor Range write total (" + std::to_string(fw.Size() >> 20) + " compressed-MiB)");
        });
        double report_time = Time() - 20, tb = Time();
        u64 const block = 1 << 26;
        for (u64 i = 0; i < fs0.size(); i += block) {
            u64 const port = std::min<u64>(block, fs0.size() - i);
            fw.Write(fs0.data() + i, port);
            if (Time() - report_time >= 30) {
                LOG << "Factor range zstd write " << std::setfill(' ') << std::setw(std::to_string(fs0.size() >> 20).size())
                    << (i >> 20) << "/" << (fs0.size() >> 20) << " M, ELA "
                    << std::fixed << std::setprecision(1) << (Time() - tb) / 60.0 << " mins" << std::endl;
                report_time = Time();
            }
        }
        fw.Flush();
    }
    {
        FileSeqReader fr(fname);
        Timing tim("Factor Range read total (<ERROR> compressed-MiB)");
        RunInDestr tim_set_name([&]{
            tim.SetName("Factor Range read total (" + std::to_string(fr.Size() >> 20) + " compressed-MiB)");
        });
        std::decay_t<decltype(fs)> buf;
        double report_time = Time() - 20, tb = Time();
        while (true) {
            buf.clear();
            buf.resize(1 << 25);
            size_t const readed = fr.Read(buf.data(), buf.size());
            ASSERT(readed <= buf.size());
            bool const last = readed < buf.size();
            buf.resize(readed);
            fs.insert(fs.end(), buf.begin(), buf.end());
            if (Time() - report_time >= 30) {
                LOG << "Factor range zstd read " << std::setfill(' ') << std::setw(std::to_string(fr.Size() >> 20).size())
                    << (fr.CReaded() >> 20) << "/" << (fr.Size() >> 20) << " MiB, ELA "
                    << std::fixed << std::setprecision(1) << (Time() - tb) / 60.0 << " mins" << std::endl;
                report_time = Time();
            }
            if (last)
                break;
        }
        ASSERT(fs.size() == (limit + 1) / 2);
    }
}

void Solve(u64 limit, size_t const L = 4) {
    size_t constexpr max_L = 4;
    u64 constexpr Mblock = MBLOCK;

    ASSERT(L <= max_L);
    
    Timing gtim("Total Solve");
    
    if (!IS_128)
        ASSERT_MSG(limit <= (1ULL << 32), "limit " + std::to_string(limit));
    
    if ((1ULL << 32) - 1 <= limit && limit <= (1ULL << 32))
        limit = (1ULL << 32) - 2;
    
    ASSERT_MSG(limit <= WordT(-1), "word bits " + std::to_string(sizeof(WordT) * 8) +
        ", limit " + std::to_string(limit));
    
    std::vector<u32> fs;
    FactorRangeCreateLoad(limit, fs);
    
    auto FName = [&](size_t l) {
        return "cpp_solutions." + std::to_string(l) + "." + std::to_string(limit);
    };
    
    struct __attribute__((packed)) Entry {
        WordT x = 0;
        bool operator == (Entry const & o) const { return x == o.x; }
    };
    
    using AType = std::vector<std::array<Entry, max_L>>;
    
    //AType A;
    
    u64 constexpr limit_start = 1, save_l_start = 3;
    size_t const start_il = 1;
    
    size_t const cpu_count = std::thread::hardware_concurrency();
    u64 A_tsize = limit - limit_start;
    
    for (size_t il = start_il; il < L; ++il) {
        size_t const l = il + 1;
        bool const il_first = il <= start_il;
        
        std::shared_ptr<FileSeqReader> fr;
        if (!il_first)
            fr = std::make_shared<FileSeqReader>(FName(l - 1) + ".raw.zst");
        FileSeqWriter fw(FName(l) + ".raw.zst");
        
        double const tb = Time();
        double report_time = -1000;
        std::atomic<double> avg_sqrs = 0, avg_sqrs_cnt = 0;
        u64 num_new_sols = 0, A_tsize2 = 0;
        
        if (l >= save_l_start) {
            std::ofstream f(FName(l));
        }
        
        for (u64 iMblock = 0; iMblock < A_tsize; iMblock += Mblock) {
            u64 const cur_Mblock = std::min<u64>(Mblock, A_tsize - iMblock);
            
            AType A(cur_Mblock);
            
            if (il_first) {
                for (u64 i = 0; i < cur_Mblock; ++i)
                    A[i][0].x = limit_start + iMblock + i;
            } else {
                ASSERT(fr);
                Timing tim("ZStd read mblock total " + std::to_string((fr->Size() * cur_Mblock / A_tsize) >> 20) + " compressed-MiB");
                u64 const rblock = 1 << 22;
                double report_time = Time() - 20;
                for (u64 i = 0; i < A.size(); i += rblock) {
                    u64 const cur_block = std::min<u64>(rblock, A.size() - i);
                    fr->Read(A.data() + i, cur_block);
                    if (Time() - report_time >= 30) {
                        LOG << "ZStd read " << (i >> 20) << "/" << (A.size() >> 20) << " M" << std::endl;
                        report_time = Time();
                    }
                }
            }
            
            size_t const nblocks = cpu_count * 4;
            u64 const block = (A.size() + nblocks - 1) / nblocks;
            std::vector<std::future<std::tuple<std::pair<u64, u64>, AType>>> asyncs;
            std::map<u64, std::tuple<std::pair<u64, u64>, AType>> Ats;
            u64 Ats_processed = 0, Ats_new = 0;
            
            for (u64 iblock = 0; iblock < A.size(); iblock += block) {
                u64 const cur_size = std::min<u64>(A.size() - iblock, block);
                
                asyncs.push_back(std::async(std::launch::async, [&, il, iblock, cur_size]{
                    AType At;
                    if (cur_size == 0)
                        return std::make_tuple(std::pair{iblock, iblock + cur_size}, std::move(At));
                    thread_local std::vector<std::tuple<u64, u64>> sqrs0;
                    auto & sqrs = sqrs0;
                    double avg_sqrs0 = 0, avg_sqrs_cnt0 = 0;
                    for (u64 i = iblock; i < iblock + cur_size; ++i) {
                        sqrs.clear();
                        FindSquares(A[i][il - 1].x, true, fs, sqrs, limit);
                        std::sort(sqrs.begin(), sqrs.end());
                        avg_sqrs0 += sqrs.size();
                        avg_sqrs_cnt0 += 1;
                        u64 const px2 = u64(A[i][il - 1].x) * A[i][il - 1].x;
                        for (auto const [y, x]: sqrs) {
                            if (y == 0)
                                continue;
                            if (x >= limit)
                                continue;
                            DWordT const x2 = DWordT(x) * x;
                            bool bad = false;
                            for (size_t jl = 0; jl + 1 < il; ++jl)
                                if (!std::get<0>(IsSquare<DWordT>(x2 - DWordT(A[i][jl].x) * A[i][jl].x))) {
                                    bad = true;
                                    break;
                                }
                            if (bad)
                                continue;
                            At.resize(At.size() + 1);
                            std::memcpy(&At.back()[0], &A[i][0], int(((u8*)&A[i][il]) - ((u8*)&A[i][0])));
                            //ASSERT_MSG(x <= u64(u32(-1)), "x_prev " + std::to_string(A[i][il - 1].x) +
                            //    ", y " + std::to_string(y) + ", x " + std::to_string(x));
                            At.back()[il].x = x;
                        }
                    }
                    avg_sqrs += avg_sqrs0;
                    avg_sqrs_cnt += avg_sqrs_cnt0;
                    return std::make_tuple(std::pair{iblock, iblock + cur_size}, AType(At));
                }));
                
                while (asyncs.size() >= cpu_count * 2 ||
                        iblock + block >= A.size() && asyncs.size() > 0) {
                    for (ptrdiff_t ie = ptrdiff_t(asyncs.size()) - 1; ie >= 0; --ie) {
                        auto & e = asyncs.at(ie);
                        if (e.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready)
                            continue;
                        auto const r = e.get();
                        asyncs.erase(asyncs.begin() + ie);
                        num_new_sols += std::get<1>(r).size();
                        Ats_new += std::get<1>(r).size();
                        Ats_processed += std::get<0>(r).second - std::get<0>(r).first;
                        Ats[std::get<0>(r).first] = std::move(r);
                    }
                    std::this_thread::yield();
                }
                
                if (Ats.size() > 0 && Time() - report_time >= 30 || iblock + block >= A.size()) {
                    double const ratio_passed = std::max<double>(1.0e-6, double(iMblock + Ats_processed) / A_tsize),
                        ratio_left = 1.0 - ratio_passed;
                    ASSERT(ratio_left >= -1.0e-6);
                    u64 A2mem = 0;
                    for (auto const & [k, v]: Ats)
                        A2mem += u64(std::get<1>(v).capacity()) * sizeof(std::decay_t<decltype(std::get<1>(v))>::value_type);
                    LOG << "L " << l << "/" << L << ", i " << std::setfill(' ') << std::setw(8)
                        << std::fixed << std::setprecision(2) << (iMblock + Ats_processed) / 1'000'000.0 << "/"
                        << std::fixed << std::setprecision(2) << A_tsize / 1'000'000.0
                        << " M, new " << std::fixed << std::setprecision(2) << (num_new_sols / 1'000'000.0) << " M, avg sqrs "
                        << std::fixed << std::setprecision(2) << (avg_sqrs / std::max<double>(1, avg_sqrs_cnt))
                        << std::setprecision(1) << ", ELA " << (Time() - tb) / 60.0 << " mins, ETA "
                        << std::setprecision(1) << (ratio_left / ratio_passed * (Time() - tb) / 60.0) << " mins, Mem [FS "
                        << (u64(fs.capacity()) * sizeof(std::decay_t<decltype(fs)>::value_type) >> 20) << " A "
                        << (u64(A.capacity()) * sizeof(std::decay_t<decltype(A)>::value_type) >> 20) << " A2 "
                        << (A2mem >> 20) << "] MiB"
                        << std::endl;
                    report_time = Time();
                }
            }
            
            ASSERT(asyncs.empty());
            
            A = AType();
            
            AType A2;
            
            {        
                Timing tim("ZStd write mblock total <ERROR> compressed-MiB");
                u64 const fw_before = fw.Size();
                RunInDestr tim_set_name([&]{
                    tim.SetName("ZStd write mblock total " + std::to_string((fw.Size() - fw_before) >> 20) +
                        " compressed-MiB"); // std::to_string((Ats_new * sizeof(AType::value_type)) >> 20)
                });
                
                double report_time = Time() - 20;
                u64 next = 0, iAts = 0;
                for (auto const [k, v]: Ats) {
                    ASSERT(k == next);
                    auto const & [a, b] = v;
                    ASSERT(k == a.first);
                    fw.Write(b.data(), b.size());
                    A_tsize2 += b.size();
                    A2.insert(A2.end(), b.begin(), b.end());
                    next = a.second;
                    if (Time() - report_time >= 30) {
                        LOG << "ZStd write " << (iAts >> 20) << "/" << (Ats_new >> 20) << " M" << std::endl;
                        report_time = Time();
                    }
                    iAts += b.size();
                }

                fw.Flush();
            }
            
            Ats = std::decay_t<decltype(Ats)>();
            
            if (l >= save_l_start) {
                std::ofstream f(FName(l), std::ios::app);
                for (auto e: A2) {
                    size_t const jend = il + 1;
                    for (size_t j = 0; j < jend; ++j)
                        f << e[j].x << (j + 1 >= jend ? "" : ", ");
                    f << std::endl;
                }
            }
        }
        LOG << std::endl << "L=" << l << " has " << std::fixed << A_tsize2 / 1'000'000 << "."
            << std::setfill('0') << std::setw(6) << (A_tsize2 % 1'000'000) << " M solutions" << std::endl;
        if (l >= save_l_start)
            LOG << "Solutions for L=" << l << " saved to '" << FName(l) << "'" << std::endl;
        LOG << std::endl;
        A_tsize = A_tsize2;
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
            LOG << "test " << i << ", " << std::flush;
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
}

int main() {
    try {
        //Test(); return 0;
        Solve(LIMIT);
        
        LogFile().close();
        return 0;
    } catch (std::exception const & ex) {
        LOG << "Exception: " << ex.what() << std::endl;
        LogFile().close();
        return -1;
    }
}