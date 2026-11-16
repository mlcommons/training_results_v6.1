#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

                                                                              
                                                                             

set -euxo pipefail

case "${CUBLAS_VERSION}-$(uname -m)" in
    13.8.1.7-aarch64) PLATFORM=sbsa SHA256=02ab2d6ed06e4da5d1540c3bd185b8b163ec401c8e9676329236ff3c7c4216be ;;
    13.8.1.7-x86_64) PLATFORM=x86_64 SHA256=1539c78dc1d2825830c7c9250f43f401c355076e8a76d959c7745bfd9ca76bb5 ;;
    *) echo "no pinned cuBLAS archive for ${CUBLAS_VERSION} on $(uname -m)" >&2; exit 1 ;;
esac

cublaslt_version() {
    python3 -c 'import ctypes; lt = ctypes.CDLL("libcublasLt.so.13"); lt.cublasLtGetVersion.restype = ctypes.c_size_t; print(lt.cublasLtGetVersion())'
}

IFS=. read -r major minor patch _ <<<"${CUBLAS_VERSION}"
wanted=$((major * 10000 + minor * 100 + patch))
base=$(cublaslt_version)
if (( base >= wanted )); then
    echo "the base image already ships cuBLAS ${base}; drop install_cublas.sh" >&2
    exit 1
fi

name=libcublas-linux-${PLATFORM}-${CUBLAS_VERSION}_cuda13-archive
curl -fsSL -o /tmp/cublas.tar.xz \
    "https://developer.download.nvidia.com/compute/cublas/redist/libcublas/linux-${PLATFORM}/${name}.tar.xz"
echo "${SHA256}  /tmp/cublas.tar.xz" | sha256sum -c -
tar -xJf /tmp/cublas.tar.xz -C /tmp --exclude='*.a'

target=/usr/local/cuda/targets/${PLATFORM}-linux
rm -f "${target}"/lib/lib{cublas,cublasLt,nvblas}.so*
cp -a "/tmp/${name}"/lib/lib{cublas,cublasLt,nvblas}.so* "${target}/lib/"
cp -a "/tmp/${name}/include/." "${target}/include/"
rm -rf /tmp/cublas.tar.xz "/tmp/${name}"
ldconfig

test "$(cublaslt_version)" = "${wanted}"
