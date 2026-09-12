#!/usr/bin/env bash
# Build the pinned Fortran oracle without writing generated files into vendor/.
#
# Example with a rootless extracted Debian toolchain:
#   FUSE_TOOLCHAIN_ROOT=/tmp/autofuse-reference-toolchain/root \
#   FUSE_REFERENCE_BUILD_DIR=/tmp/autofuse-reference-toolchain/build \
#   bash project/autofuse/build_reference.sh
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fuse_repo=${FUSE_REPO:-"$repo_root/vendor/upstream/cyrilthebault-fuse"}
toolchain_root=${FUSE_TOOLCHAIN_ROOT:?set FUSE_TOOLCHAIN_ROOT to the isolated compiler/library prefix}
out_root=${FUSE_REFERENCE_BUILD_DIR:-"$repo_root/.tmp/fuse-reference-build"}
compiler=${FUSE_REFERENCE_FC:-"$toolchain_root/usr/bin/gfortran-14"}
lib_root=${FUSE_REFERENCE_LIB_ROOT:-"$toolchain_root/lib"}
hdf5_lib_root=${FUSE_REFERENCE_HDF5_LIB_ROOT:-"$lib_root/hdf5/serial"}
include_root=${FUSE_REFERENCE_INCLUDE_ROOT:-"$toolchain_root/include"}

[[ -x "$compiler" ]] || { echo "missing Fortran compiler: $compiler" >&2; exit 2; }
[[ -f "$fuse_repo/build/Makefile" ]] || { echo "missing upstream Makefile: $fuse_repo/build/Makefile" >&2; exit 2; }
expected_commit=e6e23a4fc4ff4019bcab55f14537ea43b9525967
actual_commit=$(git -C "$fuse_repo" rev-parse HEAD)
[[ "$actual_commit" == "$expected_commit" ]] || { echo "unexpected FUSE reference commit: $actual_commit" >&2; exit 2; }
[[ -z "$(git -C "$fuse_repo" status --porcelain)" ]] || { echo "reference repository is dirty: $fuse_repo" >&2; exit 2; }
[[ -d "$lib_root" && -d "$include_root" ]] || { echo "missing toolchain library/include roots" >&2; exit 2; }

mkdir -p "$out_root/obj" "$out_root/bin"

# Debian's NetCDF C package links against these runtime libraries.  They are
# explicitly supplied so the link does not depend on --copy-dt-needed-entries.
flags="-O0 -ffree-line-length-none -fmax-errors=0 -cpp -Wl,-rpath-link,$lib_root -Wl,-rpath-link,$hdf5_lib_root -Wl,-rpath,$lib_root:$hdf5_lib_root"
libraries="-L$lib_root -L$hdf5_lib_root -lnetcdff -lnetcdf -lhdf5_hl -lhdf5 -l:libcurl-gnutls.so.4 -l:libxml2.so.2 -l:libsz.so.2 -l:libaec.so.0 -l:libcrypto.so.3 -l:libz.so.1 -l:libgnutls.so.30"

make -B -f "$fuse_repo/build/Makefile" -C "$out_root/obj" all \
  F_MASTER="$fuse_repo/" \
  FC="$compiler" \
  FLAGS="$flags" \
  FLAGS_FIXED='-O2 -c -ffixed-form' \
  LIBRARIES="$libraries" \
  NCDFF_LIB_PATH="$toolchain_root" NCDF_LIB_PATH="$toolchain_root" HDF_LIB_PATH="$toolchain_root" \
  EXE_PATH="$out_root/bin/" MOD_PATH="$out_root/obj/"

exe="$out_root/bin/fuse.exe"
[[ -x "$exe" ]] || { echo "build completed without $exe" >&2; exit 3; }
sha256sum "$exe"
"$compiler" --version | head -1
printf 'reference executable: %s\n' "$exe"
