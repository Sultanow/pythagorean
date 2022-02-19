## Arty's programm to search pythagorean triples and quadruples

On an Ubuntu machine do the following:

#### install g++-11

```console
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt upgrade
sudo apt install gcc
sudo apt install g++-11
sudo apt install libc6
sudo apt upgrade libstdc++6
```

#### install clang13

```console
sudo apt install lsb-release wget software-properties-common
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

#### install zsdt
Arty's new version allows for larger ranges as it uses additionally SSD memory.
To use it, we need to install `ZSTD` first:

```console
mkdir temp
cd temp
git clone --branch=v1.5.2 --depth=1 https://github.com/facebook/zstd
cd zstd
make -j8
make install
make check
```

#### compile
If sudo is needed `apt install sudo` followed by `sudo bash`.

For searches bigger than `2^32` we need `128 Bit` calculations. In this case we need to switch `IS_128` from `0` to `1`:

```cpp
#define IS_128 1
```

We have to add some option like "-lzstd" - this tells compiler to link ZSTD library. C++ automatically doesn't link any external libraries.

```console
g++-11 -std=c++20 -O3 -m64 prog_ssd.cpp -o prog -lpthread -lzstd -lstdc++ -lm
```

or

```console
clang-13 -std=c++20 -O3 -m64 prog_ssd.cpp -o prog -lpthread -lzstd -lstdc++ -lm
```

#### run
In order to do a conventional search up to the limit `2^29` without specifying a search interval, we call the program as follows:

```console
./prog --limit=2^29 --mblock=2^26
./prog --limit=2^32 --mblock=2^25
```

#### Explaining the limit definition
Limit is value of last element of tuple. If the limit is `1000` then `0 < w < x < y < z < 1000`.
Therefore `Solve(1 << 29)` defines a search range `0 < w < x < y < z < (1 << 29)`.

With the parameters `first_begin` and `first_end` we can specify a concrete search interval. We are searching for integers `w < x < y < z` such that `0 < first_begin <= w < first_end <= limit` and `0 < w < x < y < z < limit`. In other words we iterate all possible `w` within the intervall `[first_begin, first_end)` and the remaining integers `x`, `y`, `z` are all within the interval `[0, limit]`. By default `first_begin = 1` and `first_end = limit`.

This allows us to split tasks between PCs by splitting intervals of `w` into smaller parts. We split `[0, limit)` intervals into non-overlapping sub-intervals `[first_begin, first_end)` and distribute these sub-intervals among many PCs.

```console
./prog --limit=2^36 --mblock=2^27 --first_begin=2^35+0*2^34 --first_end=2^35+1*2^34
```

## A new memory-optimized version
The new memory-optimized version is `prog_ssd_optimized`, which consumes significantly less memory almost without loss of performance.

```console
./prog --limit=2**36 --mblock=2**25 --first_begin=0*2**34+1 --first_end=1*2**34
```

In order to lower the size of output files, we have to reduce the `MBLOCK` parameter (e.g. from `2^26` to `2^25`, which will make output files two times smaller). Both, file size and memory usage are only influenced by the `MBLOCK` parameter. The computational time is influnced only by the `first_begin` and `first_end` parameter. Using higher `limit` values will cause us to use shorter intervals. Thus the `limit` parameter influences computational time (indirectly) as well.

## Compiling and running under Windows

Install `MSYS2` and install needed tools:

```console
pacman -Syu
pacman -Su
pacman -S --needed base-devel mingw-w64-x86_64-toolchain

pacman -S mingw-w64-x86_64-gcc
pacman -S make

pacman -S mingw-w64-x86_64-zstd
```

#### compile

```console
g++ -std=c++20 -O3 -m64 /c/Users/esultano/git/pythagorean/cpp/prog_ssd.cpp -o /c/Users/esultano/git/pythagorean/cpp/prog -lpthread -lzstd -lstdc++ -lm
```

#### run

```console
/c/Users/esultano/git/pythagorean/cpp/prog.exe --limit=2**35 --mblock=2**23 --first_begin=2**34+3*2**31 --first_end=2**34+4*2**31
```

## Composing a search range by sub ranges
We can compose the search range up to `2^36` by four sub ranges each of `2^34` size, namely `[1,2^34]`, `[2^{34}+1,2*2^34]`, `[2*2^{34}+1,3*2^{34}]` and `[3*2^34+1,4*2^{34}]`. In all command lines below the `first_begin` parameter does not need to have `+1`. The program already adds `+1` by itself when needed:

```console
./prog --limit=2^36 --mblock=2^25 --first_begin=0*2^34 --first_end=1*2^34
./prog --limit=2^36 --mblock=2^25 --first_begin=1*2^34 --first_end=2*2^34
./prog --limit=2^36 --mblock=2^25 --first_begin=2*2^34 --first_end=3*2^34
./prog --limit=2^36 --mblock=2^25 --first_begin=3*2^34 --first_end=4*2^34
```
The meaning of the four parameters (command line options) `--limit`, `--mblock`, `--first_begin`, `--first_end` is such that we search through all values within `0 < w < x < y < z < limit`, where `first_begin < w < first_end`. In other words, only `w` is limited by `[first_begin, first_end]` and the remaining values `x`, `y`, `z` are limited by `[0, limit]`. In our case we have `[first_begin, first_end] = [0, 2^34]` and `[0, limit] = [0, 2^36]`, which means only `w` is below `2^34`, while all three values `x`, `y`, and `z` are below `2^36`.

## Searching almost solutions

#### compile

```console
clang-13 -std=c++20 -O3 -m64 search_almost.cpp -o search_almost -lpthread -lzstd -lstdc++ -lm
```

#### run

```console
./search_almost cpp_solutions.3.17179869184.1.17179869184 almost.txt
```

