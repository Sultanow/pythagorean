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
#include <span>

#define LIMIT (1ULL << 21)
#define IS_128 1
#define MBLOCK (1ULL << 21)

#define NUM_THREADS 0
#define FACTORS_VER 1
#define USE_POLLARD_RHO 1

#define SUPPORT_ZSTD 1
#define ZSTD_COMPRESSION_LEVEL 5
#define ZSTD_COMPR_READ_BUF_PER_CORE (1ULL << 26)
#define ZSTD_DECOMPR_READ_BUF_PER_CORE (1ULL << 27)

#define SUPPORT_ABSL 0

#if SUPPORT_ZSTD
    #include <zstd.h>
#endif

#if SUPPORT_ABSL
    #include <absl/container/flat_hash_map.h>
#endif

#define SUPPORT_POLLARD_RHO USE_POLLARD_RHO

#define ASSERT_MSG(cond, msg) { if (!(cond)) throw std::runtime_error("Assetion (" #cond ") failed at '" __FILE__ "':" + std::to_string(__LINE__) + "! Msg: '" + std::string(msg) + "'."); }
#define ASSERT(cond) ASSERT_MSG(cond, "")
#define DUMP(x) { LOG << #x << " (LN " << __LINE__ << ") = " << (x) << ", " << std::flush; }
#define LN { LOG << "LN " << __LINE__ << ", " << std::flush; }

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using u96 = unsigned _BitInt(96);
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

u64 TimeNS() {
    static auto const gtb = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
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
        : name_(name), tb_(Time()) {
        //LOG << "'" << name_ << "' time start." << std::endl;
    }
    void SetName(std::string const & name) { name_ = name; }
    ~Timing() {
        double const tp = Time() - tb_;
        LOG << "'" << name_ << "' time " << std::fixed
            << std::setprecision(2) << (tp >= 60 ? (tp / 60.0) : tp)
            << (tp >= 60 ? " mins" : " sec") << std::endl;
    }
private:
    std::string name_;
    double const tb_ = 0;
};

class BitVector {
public:
    bool Get(u64 i) const {
        return (bytes_[i / 8] >> (i % 8)) & u8(1);
    }
    u64 GetWord(u8 cnt, u64 off) const {
        return ((*(u64*)&bytes_[off / 8]) >> (off % 8)) & ((1ULL << cnt) - 1);
    }
    void SetWord(u8 cnt, u64 off, u64 word) {
        auto e = *(u64*)&bytes_[off / 8];
        e &= ~(((1ULL << cnt) - 1) << (off % 8));
        word &= (1ULL << cnt) - 1;
        word <<= off % 8;
        e |= word;
        *(u64*)&bytes_[off / 8] = e;
    }
    void SetVecElemSize(size_t cnt) {
        ASSERT_MSG(cnt <= 56, std::to_string(cnt));
        vec_el_size_ = u8(cnt);
    }
    u64 GetVec(u64 idx) const {
        return GetWord(vec_el_size_, idx * vec_el_size_);
    }
    u64 GetVecCh(u64 idx) const {
        ASSERT(idx < vec_size_);
        return GetVec(idx);
    }
    void SetVec(u64 idx, u64 word) {
        SetWord(vec_el_size_, idx * vec_el_size_, word);
    }
    void SetVecCh(u64 idx, u64 word) {
        ASSERT(idx < vec_size_);
        ASSERT(word < (1ULL << vec_el_size_));
        SetVec(idx, word);
    }
    void Set(u64 i) {
        bytes_[i / 8] |= u8(1) << (i % 8);
    }
    void ReSet(u64 i) {
        bytes_[i / 8] &= ~(u8(1) << (i % 8));
    }
    void Resize(u64 size) {
        size_ = size;
        bytes_.resize((size_ + 63) / 64 * 8);
    }
    void ResizeVec(u64 size) {
        ASSERT(vec_el_size_ > 0);
        Resize(size * vec_el_size_);
        vec_size_ = size;
    }
    u64 Size() const { return size_; }
    u64 SizeVec() const { return vec_size_; }
    u8 VecElemSize() const { return vec_el_size_; }
    std::vector<u8> & Bytes() { return bytes_; }
    std::vector<u8> const & Bytes() const { return bytes_; }
    
private:
    u64 size_ = 0, vec_size_ = 0;
    u8 vec_el_size_ = 0;
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

template <typename T>
std::string NumToStr(T x) {
    std::stringstream ss;
    auto constexpr mod = 1'000'000'000'000'000'000ULL;
    auto const hi = u64(x / mod);
    if (hi > 0)
        ss << hi;
    ss << std::setfill('0') << std::setw(std::to_string(mod - 1).size()) << u64(x % mod);
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
            T const y_start = std::max<T>(xn, 2) - 2;
            ASSERT_MSG(DT(y_start) * y_start <= S, "S " + NumToStr(S));
            for (T y = y_start + 1; y <= xn + 2; ++y)
                if (DT(y) * y > S)
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
        u64 const x64 = u64(x);
        auto const root = u64(std::sqrt(double(x64)) + 0.5);
        return std::make_tuple(root * root == x64, root);
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
        compression_level = ZSTD_COMPRESSION_LEVEL, rbuf_size0 = ZSTD_COMPR_READ_BUF_PER_CORE, wbuf_size0 = rbuf_size0;
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
    static u64 RBufSize() {
        static u64 const rbuf_size1 = std::max<size_t>(rbuf_size0, ZSTD_CStreamInSize());
        return rbuf_size1;
    }
    static u64 WBufSize() {
        static u64 const wbuf_size1 = std::max<size_t>(wbuf_size0, ZSTD_CStreamOutSize());
        return wbuf_size1;
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
        rbuf_size0 = ZSTD_DECOMPR_READ_BUF_PER_CORE, wbuf_size0 = rbuf_size0 * 3;
    using InpF = std::function<size_t(u8 * ptr, size_t size)>;
    StreamDeCompressor(InpF const & inpf)
        : rbuf_size(std::max<size_t>(rbuf_size0, ZSTD_DStreamInSize())),
          wbuf_size(std::max<size_t>(wbuf_size0, ZSTD_DStreamOutSize())),
          inpf_(inpf) {
        ctx_ = ZSTD_createDStream();
        ASSERT(ctx_);
        ZCHECK(ZSTD_initDStream(ctx_));
    }
    static u64 RBufSize() {
        static u64 const rbuf_size1 = std::max<size_t>(rbuf_size0, ZSTD_DStreamInSize());
        return rbuf_size1;
    }
    static u64 WBufSize() {
        static u64 const wbuf_size1 = std::max<size_t>(wbuf_size0, ZSTD_DStreamOutSize());
        return wbuf_size1;
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

inline size_t NThreads() {
    return NUM_THREADS == 0 ? std::thread::hardware_concurrency() : NUM_THREADS;
}

class FileSeqWriter {
public:
    using Filter = StreamCompressor;
    static u64 constexpr buf_size = 1ULL << 28;
    FileSeqWriter(std::string const & fname, size_t compression_level = size_t(-1), size_t num_threads = size_t(-1))
        : compression_level_(compression_level), num_threads_(num_threads), f_(fname, std::ios::binary) {
        if (num_threads_ == size_t(-1))
            num_threads_ = NThreads();
        ASSERT(f_.is_open());
    }
    ~FileSeqWriter() { Flush(); }
    u64 Mem() const {
        return u64(num_threads_) * (Filter::RBufSize() + Filter::WBufSize()) + buf_.capacity();
    }
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
    template <typename T>
    void Write(T const & obj) {
        Write(&obj, 1);
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
    u64 Mem() const { return Filter::RBufSize() + Filter::WBufSize(); }
    template <typename T>
    void Read(T & obj) {
        ASSERT(Read(&obj, 1) == 1);
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

std::string FloatToStr(double x, size_t round = 0) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(round) << x;
    return ss.str();
}

template <typename T> struct DWordOf;
template <> struct DWordOf<u32> : std::type_identity<u64> {};
template <> struct DWordOf<u64> : std::type_identity<u128> {};

template <typename T> struct TWordOf;
template <> struct TWordOf<u32> : std::type_identity<u96> {};
//template <> struct TWordOf<u64> : std::type_identity<u192> {};

u64 BinarySearch(u64 begin, u64 end, auto const & f) {
    while (begin < end) {
        u64 const mid = (begin + end) / 2;
        if (f(mid))
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

template <typename Word>
std::tuple<Word, size_t> BarrettRS(Word n) {
    using DWord = typename DWordOf<Word>::type;
    size_t constexpr extra = 3;

    //return std::make_tuple(Word(0), size_t(0));
    
    u64 k = BinarySearch(0, sizeof(DWord) * 8, [&](auto k){
        if (2 * (k + extra) < sizeof(Word) * 8 || (DWord(1) << k) <= DWord(n))
            return true;
        return false;
    });
    ASSERT(k < sizeof(DWord) * 8);
    
    k += extra;
    ASSERT_MSG(2 * k < sizeof(DWord) * 8, "k " + std::to_string(k));
    DWord r = (DWord(1) << (2 * k)) / n;
    ASSERT_MSG(DWord(r) < (DWord(1) << (sizeof(Word) * 8)),
        "k " + std::to_string(k) + " n " + std::to_string(n) + " r " +
        NumToStr(r) + " Word_bit_size " + std::to_string(sizeof(Word) * 8));
    ASSERT(2 * k >= sizeof(Word) * 8);
    return std::make_tuple(Word(r), size_t(2 * k - sizeof(Word) * 8));
}

template <typename Word, bool Adjust = true,
    typename DWord = typename DWordOf<Word>::type
>
Word BarrettMod(DWord const & x, Word const & n, Word const & r, size_t s) {
    using SWord = std::make_signed_t<Word>;
    
    //return x % n;
    
    DWord q;
    if constexpr(sizeof(Word) <= 4) {
        using TWord = typename TWordOf<Word>::type;
        q = DWord(((TWord(x) * r) >> (sizeof(Word) * 8)) >> s);
    } else {
        q = (
            DWord(Word(x >> (sizeof(Word) * 8))) * r +
            Word((DWord(Word(x)) * r) >> (sizeof(Word) * 8))
        ) >> s;
    }
    Word t = Word(DWord(x) - q * n);
    if constexpr(Adjust) {
        Word const mask = ~Word(SWord(t - n) >> (sizeof(Word) * 8 - 1));
        t -= mask & n;
    }
    return t;
}

template <typename A, typename B>
struct __attribute__((packed)) PackedPair {
    A first{};
    B second{};
};

class FactorRangeC {
public:
    using PrimoT = u16;
    
    #if SUPPORT_ABSL
        using Fs2PossMap = absl::flat_hash_map<i64, PackedPair<u32, u32>>;
    #else
        using Fs2PossMap = std::unordered_map<i64, PackedPair<u32, u32>>;
    #endif
    
    static bool constexpr use_map = 1;
    
    void FillPrimo() {
        primo_ps_ = std::vector<PrimoT>{2, 3, 5, 7, 11};
        primo_ = 1;
        for (auto e: primo_ps_)
            primo_ *= e;
        primo_idxs_.clear();
        primo_idxs_.resize(primo_);
        PrimoT non_div = 0;
        for (PrimoT i = 0; i < primo_idxs_.size(); ++i) {
            PrimoT divis = 0;
            for (auto e: primo_ps_)
                if (i % e == 0) {
                    divis = e;
                    break;
                }
            if (divis != 0)
                primo_idxs_[i] = divis;
            else {
                primo_idxs_[i] = non_div + 256;
                ++non_div;
            }
        }
        primo_cnt_ = non_div;
    }
    
    void Create(u64 const limit0) {
        FillPrimo();
        
        ASSERT(fs_.Size() == 0);
        
        u64 const limit_primo_cnt = (limit0 + primo_ - 1) / primo_, limit_ceil = limit_primo_cnt * primo_;
        p_bits_ = size_t(std::log2(std::max<double>(1, std::sqrt(double(limit_ceil)))) + 1);;
        
        ps_.clear();
        
        {
            Timing tim("Generate " + std::to_string(p_bits_) + "-bit primes");
            CreateLoadPrimes(p_bits_, ps_);
        }
        
        fs_.SetVecElemSize(size_t(std::log2(double(ps_.size())) + 1));
        fs_.ResizeVec(limit_primo_cnt * primo_cnt_);
        
        Timing tim("Factor Range calc total");
        double report_time = Time() - 10, tb = Time();
        u64 ip = 0;
        size_t start_ips = 0;
        for (;; ++start_ips)
            if (ps_.at(start_ips) > primo_ps_.back())
                break;
        for (size_t ip = start_ips; ip < ps_.size(); ++ip) {
            auto const p = ps_[ip];
            for (u64 i = p; i < limit_ceil; i += p) {
                PrimoT const i_primo = PrimoT(i % primo_);
                if (primo_idxs_[i_primo] < 256)
                    continue;
                /*
                ASSERT_MSG(primo_idxs_.at(i_primo) >= 256, "p " + std::to_string(p) + " primo " +
                    std::to_string(primo_) + " i " + std::to_string(i) + " i_primo " + std::to_string(i_primo) +
                    " primo_idx " + std::to_string(primo_idxs_.at(i_primo)));
                */
                u64 const idx = i / primo_ * primo_cnt_ + (primo_idxs_[i_primo] - 256);
                if (fs_.GetVecCh(idx) != 0)
                    continue;
                fs_.SetVecCh(idx, ip);
            }
            if ((ip & 0xFF) == 0 && Time() - report_time >= 30) {
                LOG << "Factor range calc " << std::setfill(' ') << std::setw(std::to_string(ps_.size() >> 10).size())
                    << (ip >> 10) << "/" << (ps_.size() >> 10) << " K, ELA "
                    << std::fixed << std::setprecision(1) << (Time() - tb) / 60.0 << " mins" << std::endl;
                report_time = Time();
            }
        }
        /*
        for (u64 i = 0; i < fs_.SizeVec(); ++i)
            if (fs_.GetVecCh(i) == 0)
                for (size_t j = 0; j < primo_idxs_.size(); ++j)
                    if (primo_idxs_[j] == i % primo_cnt_)
                        ASSERT_MSG(fs_.GetVecCh(i) != 0, "i " + std::to_string(i) + " num " + std::to_string(i / primo_cnt_ * primo_ + j));
        */
    }
    
    void CreateLoad(u64 const limit) {
        size_t constexpr compr_level = 8;
        ASSERT(fs_.Size() == 0 && ps_.size() == 0);
        std::string const fname = "factor_range.v2." + std::to_string(limit) + ".zst";
        if (!std::filesystem::exists(fname)) {
            Create(limit);
            FileSeqWriter fw(fname, compr_level);
            Timing tim("Factor Range write total (<ERROR> compressed-MiB)");
            RunInDestr tim_set_name([&]{
                tim.SetName("Factor Range write total (" + std::to_string(fw.Size() >> 20) + " compressed-MiB)");
            });
            
            fw.Write(primo_);
            fw.Write(primo_cnt_);
            fw.Write(p_bits_);
            fw.Write(ps_.size());
            fw.Write(ps_.back());
            fw.Write(primo_idxs_.size());
            fw.Write(primo_idxs_.data(), primo_idxs_.size());
            fw.Write(fs_.VecElemSize());
            fw.Write(fs_.SizeVec());
            
            double report_time = Time() - 20, tb = Time();
            u64 const block = 1ULL << 27;
            for (u64 i = 0; i < fs_.Bytes().size(); i += block) {
                u64 const port = std::min<u64>(block, fs_.Bytes().size() - i);
                fw.Write(fs_.Bytes().data() + i, port);
                if (Time() - report_time >= 30) {
                    LOG << "Factor range zstd write " << std::setfill(' ') << std::setw(std::to_string(fs_.Bytes().size() >> 20).size())
                        << (i >> 20) << "/" << (fs_.Bytes().size() >> 20) << " MiB, ELA "
                        << std::fixed << std::setprecision(1) << (Time() - tb) / 60.0 << " mins" << std::endl;
                    report_time = Time();
                }
            }
            
            fw.Flush();
        }
        size_t num_ps = 0;
        u32 last_ps = 0;
        {
            FileSeqReader fr(fname);
            Timing tim("Factor Range read total (<ERROR> compressed-MiB)");
            RunInDestr tim_set_name([&]{
                tim.SetName("Factor Range read total (" + std::to_string(fr.Size() >> 20) + " compressed-MiB)");
            });
            
            fr.Read(primo_);
            fr.Read(primo_cnt_);
            fr.Read(p_bits_);
            fr.Read(num_ps);
            fr.Read(last_ps);
            { size_t t = 0; fr.Read(t); primo_idxs_.resize(t); }
            fr.Read(primo_idxs_.data(), primo_idxs_.size());
            { u8 t = 0; fr.Read(t); fs_.SetVecElemSize(t); }
            { size_t t = 0; fr.Read(t); fs_.ResizeVec(t); }
            
            std::vector<u8> buf;
            double report_time = Time() - 20, tb = Time();
            u64 const block = 1ULL << 27;
            for (u64 i = 0; i < fs_.Bytes().size(); i += block) {
                u64 const cur_block = std::min<u64>(block, fs_.Bytes().size() - i);
                buf.clear();
                buf.resize(cur_block);
                size_t const readed = fr.Read(buf.data(), buf.size());
                ASSERT(readed == buf.size());
                std::memcpy(fs_.Bytes().data() + i, buf.data(), buf.size());
                if (Time() - report_time >= 30) {
                    LOG << "Factor range zstd read " << std::setfill(' ') << std::setw(std::to_string(fr.Size() >> 20).size())
                        << (fr.CReaded() >> 20) << "/" << (fr.Size() >> 20) << " compressed-MiB, ELA "
                        << std::fixed << std::setprecision(1) << (Time() - tb) / 60.0 << " mins" << std::endl;
                    report_time = Time();
                }
            }
        }
        {
            Timing tim("Generate " + std::to_string(p_bits_) + "-bit primes");
            CreateLoadPrimes(p_bits_, ps_);
            ASSERT(ps_.size() == num_ps);
            ASSERT(ps_.back() == last_ps);
        }
    }
    
    void FindCnts(u64 x, std::vector<std::tuple<u64, u16>> & fc) const {
        auto const ox = x;
        fc.clear();
        while (x > 1) {
            u64 p = 0;
            if ((x & 1) == 0)
                p = 2;
            else {
                auto const primo_rem = PrimoT(x % primo_);
                auto const idx = primo_idxs_[primo_rem];
                if (idx < 256)
                    p = idx;
                else {
                    auto const val = fs_.GetVecCh(x / primo_ * primo_cnt_ + (idx - 256));
                    if (val == 0)
                        p = x;
                    else
                        p = ps_.at(val);
                }
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
            ASSERT(p > 1);
            ASSERT(x % p == 0);
            x /= p;
        }
        std::sort(fc.begin(), fc.end());
        if constexpr(0) {
            u64 m = 1;
            for (auto const [k, v]: fc)
                for (size_t i = 0; i < v; ++i)
                    m *= k;
            ASSERT_MSG(ox == m, "ox " + std::to_string(ox) + " m " + std::to_string(m));
        }
    }
    
    u64 Mem() const {
        return primo_idxs_.capacity() * sizeof(std::decay_t<decltype(primo_idxs_)>::value_type) +
            fs_.Bytes().capacity() + ps_.capacity() * sizeof(std::decay_t<decltype(ps_)>::value_type) +
            ps2_.capacity() * sizeof(std::decay_t<decltype(ps2_)>::value_type) +
            fs2_vals_.capacity() * sizeof(std::decay_t<decltype(fs2_vals_)>::value_type) +
            fs2_poss_.capacity() * sizeof(std::decay_t<decltype(fs2_poss_)>::value_type) +
            fs2_poss_map_.size() * sizeof(std::decay_t<decltype(fs2_poss_map_)>::node_type) +
            brs_.capacity() * sizeof(std::decay_t<decltype(brs_)>::value_type)
            ;
    }
    
    void Factor2(std::vector<u64> nums) {
        Timing tim("Factor2 of " + FloatToStr(nums.size() / 1'000'000.0, 3) + " M nums");
        
        FillPrimo();
        
        std::sort(nums.begin(), nums.end());
        nums.resize(std::unique(nums.begin(), nums.end()) - nums.begin());
        
        {
            std::vector<u64> nums2;
            for (auto e: nums) {
                ASSERT(e >= 1);
                while (e > 1) {
                    auto const i = primo_idxs_[e % primo_];
                    if (i >= 256)
                        break;
                    ASSERT(e % i == 0);
                    e /= i;
                }
                nums2.push_back(e);
            }
            std::sort(nums2.begin(), nums2.end());
            nums2.resize(std::unique(nums2.begin(), nums2.end()) - nums2.begin());
            nums = std::move(nums2);
            nums.shrink_to_fit();
        }
        
        ASSERT(nums.size() > 0);
        
        {
            ASSERT(ps2_.empty());
            u64 const p_bits = size_t(std::log2(std::max<double>(1, std::sqrt(double(nums.back() + 1)))) + 1);;
            Timing tim("Generate " + std::to_string(p_bits) + "-bit primes");
            CreateLoadPrimes(p_bits, ps2_);
        }
        
        while (!ps2_.empty() && ps2_.front() <= primo_ps_.back())
            ps2_.erase(ps2_.begin());
        
        {
            Timing tim("Compute BarrettRS of " + FloatToStr(ps2_.size() / 1'000.0, 1) + " K primes");
            for (auto const p: ps2_)
                brs_.push_back(BarrettRS<u32>(p));
        }

        std::vector<std::vector<u64>> numsp(primo_cnt_);
        for (auto const e: nums) {
            auto i = primo_idxs_[e % primo_];
            ASSERT_MSG(i >= 256, "e " + std::to_string(e));
            numsp[i - 256].push_back(e);
        }
        
        std::vector<std::tuple<u64, u32>> all_factors;
        for (size_t i = 0; i < primo_idxs_.size(); ++i) {
            if (primo_idxs_[i] < 256)
                continue;
            auto const & e = numsp.at(primo_idxs_[i] - 256);
            if (e.empty())
                continue;
            for (size_t ip = 0; ip < ps2_.size(); ++ip) {
                //if ((ip & ((1ULL << 16) - 1)) == 0) DUMP(ip);
                auto const p = ps2_[ip];
                auto const [br, bs] = brs_.at(ip);
                u64 j = 0;
                for (j = 0;; ++j) {
                    ASSERT_MSG(j < primo_, "i " + std::to_string(i) + " p " + std::to_string(p));
                    if ((j * p) % primo_ == i)
                        break;
                }
                i64 pos = (e.front() / p + 2) * p;
                for (size_t icycle = 0;; pos -= p, ++icycle) {
                    ASSERT_MSG(icycle <= 5ULL * primo_,
                        "i " + std::to_string(i) + " p " + std::to_string(p));
                    ASSERT_MSG(i64(2ULL * p * primo_) + pos >= 0,
                        "i " + std::to_string(i) + " p " + std::to_string(p) + " e_front " +
                        std::to_string(e.front()) + " primo " + std::to_string(primo_));
                    if (pos <= i64(e.front()) && (pos + 2ULL * p * primo_) % primo_ == i)
                        break;
                }
                u64 const step = u64(p) * primo_;
                while (pos < 0)
                    pos += step;
                if constexpr(0) {
                    size_t i_e = 0;
                    for (;; pos += step) {
                        while (i_e < e.size() && e[i_e] < pos)
                            ++i_e;
                        if (i_e >= e.size())
                            break;
                        if (e[i_e] != pos)
                            continue;
                        all_factors.push_back(std::tuple{e[i_e], p});
                    }
                } else {
                    for (auto x: e) {
                        auto const mod = BarrettMod<u32, false>(x, p, br, bs);
                        if (!(mod == 0 || mod == p)) continue;
                        //if (x % p != 0) continue;
                        all_factors.push_back(std::tuple{x, p});
                    }
                }
            }
        }

        std::sort(all_factors.begin(), all_factors.end());
        
        u32 poss_prev = 0;
        if constexpr(!use_map) {
            ASSERT(fs2_poss_.empty());
            fs2_poss_.push_back(PackedPair<i64, u32>{-1, 0});
        }
        i64 prev = -1;
        for (size_t i = 0; i <= all_factors.size(); ++i) {
            u64 const k = i < all_factors.size() ? std::get<0>(all_factors[i]) : 0;
            if (i >= all_factors.size() || k != prev && prev != -1) {
                if (i >= all_factors.size())
                    prev = k;
                ASSERT(fs2_vals_.size() <= u32(-1));
                if constexpr(use_map) {
                    fs2_poss_map_[prev] = PackedPair<u32, u32>{poss_prev, u32(fs2_vals_.size())};
                    poss_prev = u32(fs2_vals_.size());
                } else
                    fs2_poss_.push_back(PackedPair<i64, u32>{prev, u32(fs2_vals_.size())});
                if (i >= all_factors.size())
                    break;
            }
            fs2_vals_.push_back(std::get<1>(all_factors[i]));
            prev = k;
        }
        fs2_poss_.shrink_to_fit();
        fs2_vals_.shrink_to_fit();
    }
    
    void FindCnts2(u64 x, std::vector<std::tuple<u64, u16>> & fc) const {
        auto const ox = x;
        ASSERT(primo_ > 0 && primo_idxs_.size() == primo_);
        fc.clear();
        auto AddF = [&](u64 p) {
            bool found = false;
            for (auto & [k, v]: fc)
                if (k == p) {
                    found = true;
                    ++v;
                    break;
                }
            if (!found)
                fc.push_back(std::make_tuple(p, 1));
        };
        while (x > 1) {
            while (x > 1) {
                auto const idx = primo_idxs_[x % primo_];
                if (idx >= 256)
                    break;
                //ASSERT(x % idx == 0);
                x /= idx;
                AddF(idx);
            }
            if (x == 1)
                break;
            size_t i_begin = 0, i_end = 0;
            if constexpr(!use_map) {
                auto it = std::lower_bound(fs2_poss_.begin(), fs2_poss_.end(),
                    PackedPair<i64, u32>{i64(x), 0U},
                        [](auto const & a, auto const & b) {
                            return a.first < b.first;
                        });
                if (it == fs2_poss_.end() || it->first != x) {
                    ASSERT(x > 1);
                    AddF(x);
                    x = 1;
                    break;
                }
                /*
                ASSERT_MSG(it != fs2_poss_.end() && it->first == x,
                    "x " + std::to_string(x));
                */
                ASSERT(it > fs2_poss_.begin());
                i_begin = (it - 1)->second;
                i_end = it->second;
            } else {
                auto const it = fs2_poss_map_.find(x);
                if (it != fs2_poss_map_.end()) {
                    i_begin = it->second.first;
                    i_end = it->second.second;
                }
            }
            for (size_t i = i_begin; i < i_end; ++i) {
                auto const d = fs2_vals_[i];
                //ASSERT(x % d == 0);
                for (size_t j = 0;; ++j) {
                    x /= d;
                    AddF(d);
                    if (x % d != 0)
                        break;
                }
            }
            if (x > 1) {
                AddF(x);
                x = 1;
                break;
            }
        }
        std::sort(fc.begin(), fc.end());
        if constexpr(0) {
            u64 m = 1;
            for (auto const [k, v]: fc)
                for (size_t i = 0; i < v; ++i)
                    m *= k;
            ASSERT_MSG(ox == m, "ox " + std::to_string(ox) + " m " + std::to_string(m));
        }
    }
    
private:
    std::vector<PrimoT> primo_ps_;
    PrimoT primo_ = 0, primo_cnt_ = 0;
    size_t p_bits_ = 0;
    std::vector<PrimoT> primo_idxs_;
    BitVector fs_;
    std::vector<u32> ps_, ps2_, fs2_vals_;
    std::vector<PackedPair<i64, u32>> fs2_poss_;
    std::vector<std::tuple<u32, size_t>> brs_;
    Fs2PossMap fs2_poss_map_;
};

#if SUPPORT_POLLARD_RHO
void FactorPollardRho(u64 N, std::vector<u64> & factors, size_t mtrials = 3, size_t trials = 16, size_t step = 0x40);
#endif

void FindSquares(u64 const N0, bool should_square, FactorRangeC const & frc, std::vector<std::tuple<u64, u64>> & sqrs, u64 limit = (u64(-1) >> 1), bool pollard_rho = false) {
    thread_local std::vector<std::tuple<u64, u16>> fc0;
    auto & fc = fc0;
    fc.clear();
    
    ASSERT(limit <= (u64(-1) >> 1));
    ASSERT(should_square);
    ASSERT_MSG(N0 <= WordT(-1), "Probably 128-bit is not enabled!");
    
    //if (should_square) ASSERT(N0 <= u32(-1));
    DWordT const N = should_square ? DWordT(N0) * N0 : N0;
    
    auto FindCntsPollardRho = [&](u64 x) {
        #if SUPPORT_POLLARD_RHO
            thread_local std::vector<u64> facts0;
            auto & facts = facts0;
            facts.clear();
            FactorPollardRho(x, facts);
            std::sort(facts.begin(), facts.end());
            u64 m = 1;
            for (auto const & f: facts) {
                m *= f;
                bool found = false;
                for (auto & [k, v]: fc)
                    if (k == f) {
                        ++v;
                        found = true;
                        break;
                    }
                if (!found)
                    fc.push_back(std::make_tuple(f, 1));
            }
            ASSERT_MSG(m == x, "m " + std::to_string(m) + " x " + std::to_string(x));
        #else
            ASSERT_MSG(false, "FindSquares: Pollard-Rho compilation is disabled!");
        #endif
    };
    
    if (!pollard_rho) {
        if constexpr(FACTORS_VER == 0)
            frc.FindCnts(N0, fc);
        else if constexpr(FACTORS_VER == 1)
            frc.FindCnts2(N0, fc);
    } else
        FindCntsPollardRho(N0);
    
    if (should_square)
        for (auto & [k, v]: fc)
            v *= 2;
    
    sqrs.clear();
    
    std::function<void(size_t, DWordT)> Iter = [&](size_t i, DWordT B){
        if (i >= fc.size()) {
            if (B > DWordT(u64(-1)))
                return;
            u64 const B64 = u64(B);
            DWordT const A = N / B64;
            ASSERT_MSG(A * B64 == N, "A " + NumToStr(A) + " B64 " +
                NumToStr(B) + " N " + NumToStr(N) + " N0 " + std::to_string(N0));
            if (A < B)
                return;
            ASSERT(((A - B64) & 1) == 0);
            DWordT const X = (A + B64) >> 1;
            if (X >= DWordT(limit))
                return;
            DWordT const Y = (A - B64) >> 1;
            if (Y == 0)
                return;
            ASSERT(N + DWordT(u64(Y)) * u64(Y) == DWordT(u64(X)) * u64(X));
            sqrs.push_back(std::make_tuple(u64(Y), u64(X)));
            return;
        }
        auto const [f, c] = fc[i];
        if (f == 2)
            B *= 2;
        for (size_t j = (f == 2 ? 1 : 0); j <= (f == 2 ? c - 1 : c); ++j) {
            Iter(i + 1, B);
            B *= f;
        }
    };
    
    Iter(0, 1);
}

void Solve(u64 limit = LIMIT, u64 first_begin = 1, u64 first_end = u64(-1), u64 const Mblock = MBLOCK, size_t const L = 4) {
    size_t constexpr max_L = 4;
    
    ASSERT(L <= max_L);
    
    Timing gtim("Total Solve");
    
    if (!IS_128)
        ASSERT_MSG(limit <= (1ULL << 32), "limit " + std::to_string(limit));
    
    if ((1ULL << 32) - 1 <= limit && limit <= (1ULL << 32))
        limit = (1ULL << 32) - 2;
    
    ASSERT_MSG(limit <= WordT(-1), "word bits " + std::to_string(sizeof(WordT) * 8) +
        ", limit " + std::to_string(limit));
    
    if (first_begin == 0)
        first_begin = 1;

    if (first_end == u64(-1))
        first_end = limit;

    ASSERT_MSG(first_begin <= first_end && first_end <= limit,
        "first_begin " + std::to_string(first_begin) + ", first_end " +
        std::to_string(first_end) + ", limit " + std::to_string(limit));
    
    LOG << "Limit " << limit << ", MBlock " << Mblock << ", FirstBegin "
        << first_begin << ", FirstEnd " << first_end << std::endl;
    
    auto FName = [&](size_t l) {
        return "cpp_solutions." + std::to_string(l) + "." + std::to_string(limit) +
            "." + std::to_string(first_begin) + "." + std::to_string(first_end);
    };
    
    struct __attribute__((packed)) Entry {
        WordT x = 0;
        bool operator == (Entry const & o) const { return x == o.x; }
    };
    
    using AType = std::vector<std::array<Entry, max_L>>;
    
    u64 constexpr save_l_start = 3;
    size_t const start_il = 1;
    
    size_t const cpu_count = NThreads();
    u64 A_tsize = first_end - first_begin;
    
    std::shared_ptr<FactorRangeC> frc;
    
    if constexpr(USE_POLLARD_RHO) {
        frc = std::make_shared<FactorRangeC>();
    } else if constexpr(FACTORS_VER == 0) {
        frc = std::make_shared<FactorRangeC>();
        frc->CreateLoad(limit);
    }
    
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
                    A[i][0].x = first_begin + iMblock + i;
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
            
            std::vector<PackedPair<WordT, u32>> sqrs_poss =
                {PackedPair<WordT, u32>{.first = 0, .second = 0}};
            std::vector<WordT> sqrs_vals;
            
            {
                std::vector<WordT> xs;
                for (auto const & e: A)
                    xs.push_back(e[il - 1].x);
                std::sort(xs.begin(), xs.end());
                xs.resize(std::unique(xs.begin(), xs.end()) - xs.begin());
                xs = std::vector<WordT>(xs);
                
                if constexpr(USE_POLLARD_RHO) {
                } else if constexpr(FACTORS_VER == 1) {
                    frc = std::make_shared<FactorRangeC>();
                    frc->Factor2(xs);
                }
                
                Timing tim("Compute unique squares " + FloatToStr(double(xs.size()) / (1 << 20), 1) +
                    " M, ratio " + FloatToStr(double(xs.size()) / A.size(), 3));
                
                size_t const nblocks = cpu_count * 4;
                u64 const block = (xs.size() + nblocks - 1) / nblocks;
                std::vector<std::future<std::tuple<u64, u64, std::decay_t<decltype(sqrs_poss)>,
                    std::decay_t<decltype(sqrs_vals)>>>> asyncs;
                
                double report_time = Time() - 20, tb = Time();
                
                for (u64 i = 0; i < xs.size(); i += block) {
                    u64 const cur_block = std::min<u64>(block, xs.size() - i);
                    asyncs.push_back(std::async(std::launch::async, [&, i, cur_block]{
                        std::decay_t<decltype(sqrs_poss)> sqrs_poss0;
                        std::decay_t<decltype(sqrs_vals)> sqrs_vals0;
                        
                        thread_local std::vector<std::tuple<u64, u64>> sqrs0;
                        auto & sqrs = sqrs0;
                        
                        ASSERT(frc);
                        for (u64 j = 0; j < cur_block; ++j) {
                            auto const & inp_x = xs[i + j];
                            sqrs.clear();
                            FindSquares(inp_x, true, *frc, sqrs, limit, USE_POLLARD_RHO);
                            std::sort(sqrs.begin(), sqrs.end());
                            for (auto const & [y, x]: sqrs) {
                                if (y == 0)
                                    continue;
                                if (x >= limit)
                                    continue;
                                sqrs_vals0.push_back(x);
                            }
                            ASSERT(sqrs_vals0.size() <= u32(-1));
                            sqrs_poss0.push_back({inp_x, u32(sqrs_vals0.size())});
                        }
                        
                        sqrs_vals0.shrink_to_fit();
                        sqrs_poss0.shrink_to_fit();
                        
                        return std::make_tuple(i, i + cur_block, std::move(sqrs_poss0), std::move(sqrs_vals0));
                    }));
                    while (asyncs.size() >= cpu_count * 2 || i + block >= xs.size() && asyncs.size() > 0) {
                        auto & e = *asyncs.begin();
                        if (e.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready) {
                            std::this_thread::yield();
                            continue;
                        }
                        auto const res = e.get();
                        auto const & [begin, end, poss, vals] = res;
                        u64 prev_off = 0;
                        for (auto const & [inp_x, off]: poss) {
                            for (u64 j = prev_off; j < off; ++j)
                                sqrs_vals.push_back(vals.at(j));
                            ASSERT(sqrs_vals.size() <= u32(-1));
                            sqrs_poss.push_back({inp_x, u32(sqrs_vals.size())});
                            prev_off = off;
                        }
                        asyncs.erase(asyncs.begin());
                        if (Time() - report_time >= 30
                            // || asyncs.size() == 0
                        ) {
                            LOG << "Unique squares i " << std::fixed
                                << std::setprecision(1) << double(sqrs_poss.size() - 1) / (1 << 20) << "/"
                                << std::setprecision(1) << double(xs.size()) / (1 << 20) << " M, ELA "
                                << std::setprecision(1) << (Time() - tb) / 60.0 << " mins, ETA "
                                << std::setprecision(1) << (1.0 - double(sqrs_poss.size() - 1) / xs.size()) * (Time() - tb) /
                                    (double(sqrs_poss.size() - 1) / xs.size()) / 60.0 << " mins" << std::endl;
                            report_time = Time();
                        }
                    }
                }
                
                sqrs_poss.shrink_to_fit();
                sqrs_vals.shrink_to_fit();
            }
            
            size_t const nblocks = cpu_count * 4;
            u64 const block = (A.size() + nblocks - 1) / nblocks;
            std::vector<std::future<std::tuple<std::pair<u64, u64>, AType>>> asyncs;
            std::map<u64, std::tuple<std::pair<u64, u64>, AType>> Ats;
            u64 Ats_processed = 0, Ats_new = 0;
            
            {
                Timing tim("Composing mblock tuples from unique squares");
                
                std::atomic<u64> div_filt = 0, div_filt_total = 0;
                
                for (u64 iblock = 0; iblock < A.size(); iblock += block) {
                    u64 const cur_size = std::min<u64>(A.size() - iblock, block);
                    
                    asyncs.push_back(std::async(std::launch::async, [&, il, iblock, cur_size]{
                        AType At;
                        if (cur_size == 0)
                            return std::make_tuple(std::pair{iblock, iblock + cur_size}, std::move(At));
                        double avg_sqrs0 = 0, avg_sqrs_cnt0 = 0;
                        for (u64 i = iblock; i < iblock + cur_size; ++i) {
                            auto const inp_x = A[i][il - 1].x;
                            auto const it_second = std::lower_bound(sqrs_poss.begin() + 1, sqrs_poss.end(), PackedPair<WordT, u32>{
                                inp_x, u32(0)}, [&](auto const & a, auto const & b){ return a.first < b.first; });
                            auto const it_first = it_second - 1;
                            ASSERT(sqrs_poss.begin() <= it_second && it_second < sqrs_poss.end());
                            ASSERT(it_second->first == inp_x);
                            std::span<WordT> sqrs(sqrs_vals.begin() + (it_first->second), sqrs_vals.begin() + (it_second->second));
                            avg_sqrs0 += sqrs.size();
                            avg_sqrs_cnt0 += 1;
                            for (auto const & x: sqrs) {
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
                                if (0 && l == 2) {
                                    size_t d0 = 0, d1 = 0;
                                    for (; (A[i][0].x & ((1ULL << (d0 + 1)) - 1)) == 0; ++d0);
                                    for (; (x & ((1ULL << (d1 + 1)) - 1)) == 0; ++d1);
                                    ++div_filt_total;
                                    if (d1 > d0 || d1 + 1 == d0) {
                                        ++div_filt;
                                        continue;
                                    }
                                }
                                At.resize(At.size() + 1);
                                std::memcpy(&At.back()[0], &A[i][0], int(((u8*)&A[i][il]) - ((u8*)&A[i][0])));
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
                            << std::setprecision(1) << (ratio_left / ratio_passed * (Time() - tb) / 60.0) << " mins, Mem [USqrs "
                            << ((sqrs_poss.capacity() * sizeof(std::decay_t<decltype(sqrs_poss)>::value_type) +
                                 sqrs_vals.capacity() * sizeof(std::decay_t<decltype(sqrs_vals)>::value_type)) >> 20)
                            << " ZStd " << ((fw.Mem() + fr->Mem()) >> 20) << " FS " << (frc->Mem() >> 20) << " A "
                            << (u64(A.capacity()) * sizeof(std::decay_t<decltype(A)>::value_type) >> 20) << " A2 "
                            << (A2mem >> 20) << "] MiB"
                            << std::endl;
                        report_time = Time();
                    }
                }
                
                if (div_filt_total > 0)
                    LOG << "DivFilt " << div_filt << "/" << div_filt_total << std::endl;
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
                {
                    Timing tim("Saving mblock results to file '" + FName(l) + "'");
                    std::ofstream f(FName(l), std::ios::app);
                    for (auto const & e: A2) {
                        size_t const jend = il + 1;
                        for (size_t j = 0; j < jend; ++j)
                            f << e[j].x << (j + 1 >= jend ? "" : ", ");
                        f << std::endl;
                    }
                }
                {
                    Timing tim("Checking mblock tuples correctness");
                    for (auto const & e: A2)
                        for (size_t jl = 0; jl <= il; ++jl) {
                            ASSERT(std::get<0>(IsSquare<DWordT>(DWordT(e[jl].x) * e[jl].x)));
                            for (size_t kl = 0; kl < jl; ++kl)
                                ASSERT(std::get<0>(IsSquare<DWordT>(DWordT(e[jl].x) * e[jl].x - DWordT(e[kl].x) * e[kl].x)));
                        }
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
    
    FactorRangeC frc;
    frc.CreateLoad(limit);
    
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
        FindSquares(N, true, frc, sqrs0);
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

#if SUPPORT_POLLARD_RHO

u64 RandomU64() {
    thread_local std::mt19937_64 rng{(u64(std::random_device{}()) << 32) | u64(std::random_device{}())};
    return rng();
}

u64 PowMod(u64 a, u64 b, u64 const c) {
    u64 r = 1;
    while (b != 0) {
        if (b & 1)
            r = (u128(r) * a) % c;
        a = (u128(a) * a) % c;
        b >>= 1;
    }
    return r;
}

bool IsFermatPrp(u64 N, size_t ntrials = 32) {
    // https://en.wikipedia.org/wiki/Fermat_primality_test
    if (N <= 16)
        return N == 2 || N == 3 || N == 5 || N == 7 || N == 11 || N == 13;
    for (size_t trial = 0; trial < ntrials; ++trial) {
        u64 const witness = RandomU64() % (N - 3) + 2;
        if (PowMod(witness, N - 1, N) != 1) {
            //LOG << "FermatPrp N " << N << " witness " << witness << " powmod " << PowMod(witness, N - 1, N) << std::endl;
            return false;
        }
    }
    return true;
}

u64 GCD(u64 a, u64 b) {
    while (b != 0)
        std::tie(a, b) = std::make_tuple(b, a % b);
    return a;
}

u64 FactorTrialDivision(u64 N, std::vector<u64> & factors, u64 limit = u64(-1) >> 1) {
    // https://en.wikipedia.org/wiki/Trial_division
    if (N <= 1)
        return N;
    while ((N & 1) == 0) {
        factors.push_back(2);
        N >>= 1;
    }
    bool checked_all = false;
    for (u64 d = 3;; d += 2) {
        if (d * d > N) {
            checked_all = true;
            break;
        }
        if (d > limit)
            break;
        while (N % d == 0) {
            factors.push_back(d);
            N /= d;
        }
    }
    if (N > 1 && checked_all) {
        factors.push_back(N);
        N = 1;
    }
    return N;
}

void FactorPollardRhoV0(u64 N, std::vector<u64> & factors) {
    // https://en.wikipedia.org/wiki/Pollard%27s_rho_algorithm
    
    u64 tdiv_limit = 1ULL << 3;
    
    for (size_t mtrial = 0; mtrial < 3; ++mtrial) {
        tdiv_limit *= tdiv_limit;
        N = FactorTrialDivision(N, factors, tdiv_limit);
        
        if (N <= 1)
            return;
        
        if (IsFermatPrp(N)) {
            factors.push_back(N);
            return;
        }
        
        auto f = [&N](auto x) -> u64 { return (u128(x + 1) * (x + 1)) % N; };
        auto DiffAbs = [](auto x, auto y){ return x >= y ? x - y : y - x; };
        
        for (size_t trial = 0; trial < 16; ++trial) {
            u64 x = RandomU64() % (N - 3) + 1;
            size_t total_steps = 0;
            for (size_t cycle = 1;; ++cycle) {
                bool good = true;
                u64 y = x;
                for (u64 i = 0; i < (u64(1) << cycle); ++i) {
                    x = f(x);
                    ++total_steps;
                    u64 const d = GCD(DiffAbs(x, y), N);
                    if (d > 1) {
                        if (d == N) {
                            good = false;
                            break;
                        }
                        //std::cout << N << ": " << d << ", " << total_steps << std::endl;
                        ASSERT(N % d == 0);
                        FactorPollardRhoV0(d, factors);
                        FactorPollardRhoV0(N / d, factors);
                        return;
                    }
                }
                if (!good)
                    break;
            }
        }
    }
    
    ASSERT_MSG(false, "Pollard Rho factorization failed for N = " + std::to_string(N));
}

void FactorPollardRho(u64 N, std::vector<u64> & factors, size_t mtrials, size_t trials, size_t step) {
    // https://en.wikipedia.org/wiki/Pollard%27s_rho_algorithm
    
    u64 tdiv_limit = 1ULL << 3;
    
    for (size_t mtrial = 0; mtrial < mtrials; ++mtrial) {
        tdiv_limit *= tdiv_limit;
        N = FactorTrialDivision(N, factors, tdiv_limit);
        
        if (N <= 1)
            return;
        
        if (IsFermatPrp(N)) {
            factors.push_back(N);
            return;
        }
        
        u64 br = 0, bs = 0;
        std::tie(br, bs) = BarrettRS<u64>(N);
        
        auto f = [&](auto x) -> u64 { return BarrettMod<u64, false>(u128(x + 1) * (x + 1), N, br, bs); };
        auto DiffAbs = [](auto x, auto y){ return x >= y ? x - y : y - x; };
        
        for (size_t trial = 0; trial < trials; ++trial) {
            u64 x = RandomU64() % (N - 3) + 1;
            size_t total_steps = 0;
            for (size_t cycle = 1;; ++cycle) {
                bool good = true;
                u64 y = x, i_start = 0, i_hi = (u64(1) << cycle), x_start = x, mod = 1;
                for (u64 i = 0; i < i_hi; ++i) {
                    x = f(x);
                    ++total_steps;
                    mod = BarrettMod<u64, false>(u128(N + x - y) * mod, N, br, bs);
                    if (!(i - i_start >= step || i + 1 >= i_hi || mod == 0 || mod == N))
                        continue;
                    u64 const gcd = GCD(mod, N);
                    if (gcd == 1) {
                        x_start = x;
                        i_start = i + 1;
                        continue;
                    }
                    u64 x2 = x_start;
                    for (u64 j = i_start; j <= i; ++j) {
                        x2 = f(x2);
                        u64 const d = GCD(N + x2 - y, N);
                        if (d <= 1)
                            continue;
                        if (d == N) {
                            good = false;
                            break;
                        }
                        ASSERT(N % d == 0);
                        FactorPollardRho(d, factors, mtrials, trials, step);
                        FactorPollardRho(N / d, factors, mtrials, trials, step);
                        return;
                    }
                    x_start = x;
                    i_start = i + 1;
                    if (!good)
                        break;
                    ASSERT(false);
                }
                if (!good)
                    break;
            }
        }
    }
    
    ASSERT_MSG(false, "Pollard Rho factorization failed for N = " + std::to_string(N));
}

#if 0
void TestPollard() {
    ASSERT(IS_128);
    ASSERT(IsFermatPrp(8'072'791));
    {
        std::vector<u64> fs;
        FactorPollardRho(37'900'949, fs);
    }
    
    for (size_t bits = 8; bits <= 52; bits += 4) {
        double const tb = Time();
        LOG << "test_bits " << bits << " (";
        auto const hi = (bits <= 48 ? (1 << 7) : (1 << 4));
        for (size_t itest = 0; itest < hi; ++itest) {
            u64 const N = RandomU64() % (1ULL << bits);
            {
                auto const [is_sqr, root] = IsSquare<u128>(u128(N) * N);
                ASSERT(is_sqr);
                ASSERT(u128(root) * root == u128(N) * N);
            }
            {
                auto const [is_sqr, root] = IsSquare<u128>(N - N % 4 + 2);
                ASSERT(!is_sqr);
            }
            std::vector<u32> fs_dummy;
            std::vector<std::tuple<u64, u64>> sqrs;
            FindSquares(N, true, fs_dummy, sqrs, (u64(-1) >> 1), true);
            for (auto const [y, x]: sqrs) {
                //LOG << "y = " << y << ", x = " << x << std::endl;
                ASSERT(u128(N) * N + u128(y) * y == u128(x) * x);
            }
        }
        LOG << std::fixed << std::setprecision(3) << (Time() - tb) << " sec), ";
    }
}
#endif

void TestPollard2() {
    {
        auto const N = 1415926535897932ULL;
        std::vector<u64> fs;
        FactorPollardRho(N, fs);
        std::sort(fs.begin(), fs.end());
        std::string s;
        for (auto x: fs)
            s += std::to_string(x) + ", ";
        ASSERT_MSG(s == "2, 2, 127, 137, 13217, 1539301, ", s);
    }
    auto ToStr = [](auto const & v){
        std::stringstream ss;
        for (auto x: v)
            ss << x << ", ";
        return ss.str();
    };
    {
        for (size_t itest = 0; itest < (1 << 13); ++itest) {
            std::vector<u64> fs, fs2;
            u64 n = 1;
            for (size_t i = 0; i < 3; ++i) {
                u64 n0 = RandomU64() % (1 << 19) + 2;
                FactorTrialDivision(n0, fs);
                n *= n0;
            }
            FactorPollardRho(n, fs2, 1);
            std::sort(fs.begin(), fs.end());
            std::sort(fs2.begin(), fs2.end());
            ASSERT_MSG(fs == fs2, "n " + std::to_string(n) + " | " +
                ToStr(fs) + " | " + ToStr(fs2));
        }
    }
    {
        std::vector<u64> nums;
        for (size_t itest = 0; itest < (1 << 12); ++itest) {
            u64 n = 1;
            for (size_t i = 0; i < 2; ++i)
                n *= RandomU64() % (1 << 28) + 2;
            nums.push_back(n);
        }
        std::vector<u64> fs0, fs1;
        {
            auto tim0 = TimeNS();
            for (auto n: nums) {
                fs0.clear();
                FactorPollardRhoV0(n, fs0);
            }
            tim0 = TimeNS() - tim0;
            for (auto step: {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200}) {
                auto tim1 = TimeNS();
                for (auto n: nums) {
                    fs1.clear();
                    FactorPollardRho(n, fs1, 3, 16, step);
                }
                tim1 = TimeNS() - tim1;
                LOG << "Pollard V0 speed " << (tim0 / nums.size() / 1000.0) << " mcs/num, V1 (step 0x"
                    << std::hex << step << std::dec << ") speed " << (tim1 / nums.size() / 1000.0)
                    << " mcs/num, boost " << std::fixed << std::setprecision(4) << double(tim0) / tim1 << std::endl;
            }
            for (auto n: nums) {
                fs0.clear();
                FactorPollardRhoV0(n, fs0);
                fs1.clear();
                FactorPollardRho(n, fs1);
                std::sort(fs0.begin(), fs0.end());
                std::sort(fs1.begin(), fs1.end());
                ASSERT_MSG(fs0 == fs1, "N " + std::to_string(n) + " fs0 " + ToStr(fs0) + " fs1 " + ToStr(fs1));
            }
        }
    }
}

void TestBinarySearch() {
    size_t was_end = 0;
    for (size_t itest = 0; (itest < (1 << 16)) || (was_end < 10); ++itest) {
        u64 const end = RandomU64() % ((1 << 7) + 5),
            begin = std::min<u64>(RandomU64() % ((1 << 7) + 5), end),
            mid = begin + RandomU64() % (end - begin + 1);
        if (mid == end)
            ++was_end;
        auto f = [mid](u64 i){ return i < mid; };
        ASSERT(BinarySearch(begin, end, f) == mid);
    }
}

#endif

auto ParseProgOpts(int argc, char ** argv) {
    std::vector<std::string> args;
    for (size_t i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);
    std::map<std::string, std::string> m;
    for (auto e: args) {
        auto const oe = e;
        ASSERT_MSG(e.size() >= 2 && e[0] == '-' && e[1] == '-', oe);
        e.erase(0, 2);
        auto const pos = e.find('=');
        ASSERT_MSG(pos != std::string::npos, oe);
        std::string key;
        if (pos != std::string::npos) {
            key = e.substr(0, pos);
            e.erase(0, pos + 1);
        }
        std::string val = e;
        m[key] = val;
    }
    return m;
}

i64 StrToNum(std::string const & s) {
    std::function<i64(std::string const &)> Expr = [&](std::string const & s) -> i64 {
        auto pos = s.find(' ');
        if (pos != std::string::npos)
            return Expr(s.substr(0, pos) + s.substr(pos + 1));
        ASSERT(!s.empty());
        pos = s.find('+');
        if (pos != std::string::npos)
            return Expr(s.substr(0, pos)) + Expr(s.substr(pos + 1));
        pos = s.rfind('-');
        if (pos != std::string::npos)
            return Expr(s.substr(0, pos)) - Expr(s.substr(pos + 1));
        pos = s.find('*');
        if (pos != std::string::npos && (pos + 1 >= s.size() || s[pos + 1] != '*'))
            return Expr(s.substr(0, pos)) * Expr(s.substr(pos + 1));
        pos = s.find('/');
        if (pos != std::string::npos)
            return Expr(s.substr(0, pos)) / Expr(s.substr(pos + 1));
        pos = s.find('^');
        if (pos != std::string::npos)
            return std::llround(std::pow(double(Expr(s.substr(0, pos))), double(Expr(s.substr(pos + 1)))));
        pos = s.find("**");
        if (pos != std::string::npos)
            return std::llround(std::pow(double(Expr(s.substr(0, pos))), double(Expr(s.substr(pos + 2)))));
        return std::stoll(s);
    };
    return Expr(s);
}

static std::map<std::string, std::string> PO;

int main(int argc, char ** argv) {
    try {
        //TestPollard2(); return 0;
        PO = ParseProgOpts(argc, argv);
        Solve(
            PO.count("limit") ? StrToNum(PO.at("limit")) : LIMIT,
            PO.count("first_begin") ? StrToNum(PO.at("first_begin")) : 1,
            PO.count("first_end") ? u64(StrToNum(PO.at("first_end"))) : u64(-1),
            PO.count("mblock") ? StrToNum(PO.at("mblock")) : MBLOCK
        );
        
        LogFile().close();
        return 0;
    } catch (std::exception const & ex) {
        LOG << "Exception: " << ex.what() << std::endl;
        LogFile().close();
        return -1;
    }
}