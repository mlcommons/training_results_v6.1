#!/usr/bin/env python3
"""build_lmh.py <src dir> <build dir> <ext dir>: builds ak_l8b_lmh.so (ak_l8b_lmh.cpp, linked to cuBLASLt) in the image
and copies it to <ext dir>/ak_l8b_lmh.so; prints its sha256."""
import argparse
import hashlib
import os
import shutil

from torch.utils.cpp_extension import load

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("src")
_ap.add_argument("build")
_ap.add_argument("ext")
_args = _ap.parse_args()
src, build, ext = _args.src, _args.build, _args.ext
os.makedirs(build, exist_ok=True)
os.makedirs(ext, exist_ok=True)
m = load(name="ak_l8b_lmh", sources=[os.path.join(src, "ak_l8b_lmh.cpp")], extra_ldflags=["-lcublasLt"],
         extra_cflags=["-O2"], build_directory=build, with_cuda=True, verbose=False)
so = m.__file__
shutil.copyfile(so, os.path.join(ext, "ak_l8b_lmh.so"))
print("[build] %s -> %s sha256 %s" % (so, os.path.join(ext, "ak_l8b_lmh.so"),
                                       hashlib.sha256(open(so, "rb").read()).hexdigest()))
