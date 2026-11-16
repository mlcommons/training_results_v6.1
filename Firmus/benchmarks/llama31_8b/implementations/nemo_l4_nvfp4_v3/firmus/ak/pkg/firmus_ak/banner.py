"""Firmus L4: prove, from inside the training run, that the kernels running are the ones this image was built with.

Imported by firmus_ak/__init__.py before any patch module (and so before torch, Transformer Engine or Megatron are
imported by the recipe).  On EVERY rank it re-hashes

  * every file the image build recorded under ${FIRMUS_AK_PREFIX} (/FIRMUS_AK_BUILD_SHA256.txt: the compiled extensions,
    the Triton kernel sources, the pinned LM-head algorithms), and checks that no other .so / .py file was added there;
  * every upstream file the Firmus patches changed, as installed (/FIRMUS_STACK_SHA256.txt: Transformer Engine's rope.py
    and dot_product_attention/utils.py, the five Megatron-Core files, the recipe's custom_embedding.py and pretrain.py);

and raises RuntimeError on any difference, a missing file or a missing record (fail loud, before the model is built).
Rank 0 prints, for the post-run checks (post_check.sh: "auto_kernel banner] MATCH" and "tree_sha256=<TREE_SHA>"):

    [auto_kernel banner] package=/opt/firmus/ak files=<n> stack_files=<m>
    [auto_kernel banner] tree_sha256=<sha256 of /FIRMUS_AK_BUILD_SHA256.txt>  (= FIRMUS_AK_BUILD_SHA256 in /FIRMUS_AK.env)
    [auto_kernel banner] stack_sha256=<sha256 of /FIRMUS_STACK_SHA256.txt>
    [auto_kernel banner] MATCH

It reads files only; it imports nothing beyond the standard library.
"""

import hashlib
import os

PREFIX = "[auto_kernel banner]"
BUILD_RECORD = "/FIRMUS_AK_BUILD_SHA256.txt"
STACK_RECORD = "/FIRMUS_STACK_SHA256.txt"


def _rank():
    for var in ("SLURM_PROCID", "RANK", "OMPI_COMM_WORLD_RANK"):
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return 0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _records(path):
    """sha256sum-format lines -> [(sha, path)]; raises if the record is missing or malformed."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            sha, sep, name = line.partition("  ")
            if not sep or len(sha) != 64:
                raise RuntimeError("{} {} line {} is not a sha256sum line: {!r}".format(PREFIX, path, n, line))
            out.append((sha, name))
    if not out:
        raise RuntimeError("{} {} is empty".format(PREFIX, path))
    return out


def check():
    prefix = os.environ.get("FIRMUS_AK_PREFIX", "/opt/firmus/ak")
    bad = []
    built = _records(BUILD_RECORD)
    listed = set()
    for sha, name in built:
        p = os.path.normpath(os.path.join(prefix, name))
        listed.add(p)
        if not os.path.isfile(p):
            bad.append("missing " + p)
        elif _sha256(p) != sha:
            bad.append("changed " + p)
    for root, dirs, files in os.walk(prefix):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            p = os.path.normpath(os.path.join(root, fn))
            if fn.endswith((".so", ".py", ".json")) and p not in listed:
                bad.append("not built by the image: " + p)
    stack = _records(STACK_RECORD)
    for sha, name in stack:
        if not os.path.isfile(name):
            bad.append("missing " + name)
        elif _sha256(name) != sha:
            bad.append("changed " + name)
    if bad:
        raise RuntimeError("{} MISMATCH on rank {}: {} problem(s): {}".format(PREFIX, _rank(), len(bad), "; ".join(bad[:8])))
    lines = ["package={} files={} stack_files={}".format(prefix, len(built), len(stack)),
             "tree_sha256={}".format(_sha256(BUILD_RECORD)),
             "stack_sha256={}".format(_sha256(STACK_RECORD)),
             "MATCH"]
    if _rank() == 0:
        for line in lines:
            print("{} {}".format(PREFIX, line), flush=True)
    return lines


check()
