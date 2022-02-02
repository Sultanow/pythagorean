## Arty's programm to search pythagorean triples and quadruples

On an Ubuntu machine do the following:

install g++-11

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

install clang13:

```console
sudo apt install lsb-release wget software-properties-common
sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

compile

```console
g++-11 -std=c++20 -O3 -m64 prog.cpp -o prog -lpthread
```

or

```console
clang-13 -std=c++20 -O3 -m64 prog.cpp -o prog -lpthread
```

run it

```console
./prog
```

## Version for larger ranges

Arty's new version `prog_ssd.cpp` allows for larger ranges as it uses additionally SSD memory.
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

We have to add some option like "-lzstd" - this tells compiler to link ZSTD library. C++ automatically doesn't link any external libraries.

run it using:

```console
clang-13 -std=c++20 -O3 -m64 prog_ssd.cpp -o prog -lpthread -lzstd -lstdc++ -lm
```

If sudo is needed `apt install sudo` followed by `sudo bash`.

For searches bigger than `2^32` we need 128 Bit calculations. In this case we need to switch `IS_128` from `0` to `1`:

```cpp
#define IS_128 1
```

## Explaining the limit definition
Limit is value of last element of tuple. If the limit is `1000` then `0 < w < x < y < z < 1000`.
Therefore `Solve(1 << 29)` defines a search range `0 < w < x < y < z < (1 << 29)`.

## Splitting and Resuming
With the parameters `first_begin` and `first_end` we can specify a concrete search interval. We are searching for integers `w < x < y < z` such that `0 < first_begin <= w < first_end <= limit` and `0 < w < x < y < z < limit`. In other words we iterate all possible `w` within the intervall `[first_begin, first_end)` and the remaining integers `x`, `y`, `z` are all within the interval `[0, limit]`. By default `first_begin = 1` and `first_end = limit`.

This allows us to split tasks between PCs by splitting intervals of `w` into smaller parts. We split `[0, limit)` intervals into non-overlapping sub-intervals `[first_begin, first_end)` and distribute these sub-intervals among many PCs.

```console
./prog --limit=2^36 --mblock=2^27 --first_begin=2^35+0*2^34 --first_end=2^35+1*2^34
```

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

compile it via:
```console
g++ -std=c++20 -O3 -m64 /c/Users/esultano/git/pythagorean/cpp/prog_ssd.cpp -o /c/Users/esultano/git/pythagorean/cpp/prog -lpthread -lzstd -lstdc++ -lm
```

run it via:
```console
/c/Users/esultano/git/pythagorean/cpp/prog.exe --limit=2**35 --mblock=2**23 --first_begin=2**34+3*2**31 --first_end=2**34+4*2**31
```
