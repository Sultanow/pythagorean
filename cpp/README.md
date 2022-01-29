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