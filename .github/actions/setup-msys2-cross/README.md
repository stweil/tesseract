# Set up MSYS2 cross build

This action prepares a Debian or Ubuntu runner for cross builds for Windows
which use the libraries of [MSYS2](https://www.msys2.org/):

- [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) provides the cross
  compiler (clang), the linker (lld) and the other cross tools.
- `pacman` installs the C runtime, the C++ standard library and the
  requested libraries from the MSYS2 package repositories.

The MSYS2 packages are installed in the same directory as on Windows, for
example `/clang64`. That directory is the sysroot of the cross compiler, so a
cross build uses the same headers, import libraries and DLL files as a native
build in that MSYS2 environment. No package of the Linux distribution
provides code for Windows.

## Usage

```yaml
- uses: actions/checkout@v7
- id: cross
  uses: ./.github/actions/setup-msys2-cross
  with:
    msystem: CLANG64
    packages: leptonica libarchive pango
- run: |
    ./configure --host=${{ steps.cross.outputs.host }}
    make
```

Other repositories can use
`tesseract-ocr/tesseract/.github/actions/setup-msys2-cross@main`.

## Inputs

| Name | Default | Description |
| --- | --- | --- |
| `msystem` | `CLANG64` | MSYS2 environment: `CLANG64`, `CLANGARM64` or `UCRT64` |
| `packages` | | MSYS2 packages without the prefix of the environment, separated by white space |
| `llvm-mingw-version` | see `setup.sh` | release of llvm-mingw |
| `llvm-mingw-sha256` | | SHA-256 checksum of the llvm-mingw archive; the default release is always checked |

## Outputs

| Name | Example | Description |
| --- | --- | --- |
| `host` | `x86_64-w64-mingw32` | host triplet for the cross build |
| `prefix` | `/clang64` | installation directory of the MSYS2 packages |
| `package-prefix` | `mingw-w64-clang-x86_64` | prefix of the MSYS2 package names |
| `path` | | directories with the cross tools; the action adds them to `PATH` |

## Cross tools

The action adds two directories to `PATH`:

- `/opt/msys2-cross/<environment>/bin` with wrapper scripts
  `<host>-clang`, `<host>-clang++`, `<host>-gcc`, `<host>-g++`, `<host>-cc`
  and `<host>-c++`, which run the compiler of llvm-mingw with the MSYS2
  environment as sysroot, and `<host>-pkg-config`, which only finds the
  packages of that environment.
- `/opt/llvm-mingw/bin` with the other tools (`<host>-ar`, `<host>-strip`,
  `<host>-windres`, ...).

Autotools find all of them with `--host=<host>`.

## MSYS2 environments

| Environment | C++ standard library | Runtime libraries which the result needs |
| --- | --- | --- |
| `CLANG64`, `CLANGARM64` | libc++ | `libc++.dll`, sometimes `libunwind.dll` |
| `UCRT64` | libstdc++ | `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll` |

All of them use the Universal C Runtime (UCRT). `CLANG64` is the environment
which matches llvm-mingw. For `UCRT64` the compiler uses libstdc++ and libgcc
from the GCC package of MSYS2, so that C++ libraries like ICU can be used.
The `MINGW64` environment is deprecated by MSYS2 and not supported.

## Use without GitHub Actions

`setup.sh` can also be run directly, for example

    .github/actions/setup-msys2-cross/setup.sh UCRT64 leptonica pango

It needs root rights (it uses `sudo` unless it is run by root) and prints
the directories which must be added to `PATH`.
