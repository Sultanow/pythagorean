###Arty's programm to search pythagorean triples and quadruples

On an Ubuntu machine do the following:

install g++-11

```console
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-11
```

install clang13 (possibly a `sudo apt install lsb-release wget software-properties-common` is required before)

```console
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

For searches bigger than `2^32` we need 128 Bit calculations. In this case we need to switch `IS_128` from `0` to `1`:

```cpp
#define IS_128 1
```
