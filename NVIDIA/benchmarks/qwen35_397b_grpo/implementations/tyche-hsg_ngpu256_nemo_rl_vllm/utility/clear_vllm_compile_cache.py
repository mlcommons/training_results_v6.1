# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clear only container-private vLLM compilation artifacts between experiments.

Call after the previous experiment's workers have stopped, on every allocated
node. There is deliberately no command-line path override. Fixture paths are
accepted by the Python function so tests never need to touch /root.
"""

import os
from pathlib import Path
import re
import shutil
import socket
import stat
import sys
from typing import Mapping, Sequence


CACHE_PARENT = Path("/root/.cache")
_CACHE_NAME = re.compile(r"vllm(?:_[0-9]+)?\Z")


class CacheCleanupError(RuntimeError):
    """A cache boundary could not be established safely."""


def _check_environment(environ: Mapping[str, str]) -> None:
    expected = {
        "HOME": "/root",
        "XDG_CACHE_HOME": "/root/.cache",
        "VLLM_CACHE_ROOT": "/root/.cache/vllm",
    }
    for key, value in expected.items():
        if key == "HOME" or key in environ:
            if environ.get(key) != value:
                raise CacheCleanupError(f"unsupported {key}; refusing cache cleanup")


def _directory_exists(path: Path) -> bool:
    """Reject symlinks and non-directories anywhere in an absolute path."""
    if not path.is_absolute() or path.resolve(strict=False) != path:
        raise CacheCleanupError(f"non-canonical cache path: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(mode):
            raise CacheCleanupError(f"cache ancestor/target is not a plain directory: {current}")
    return True


def _mountpoints(mountinfo: str) -> Sequence[Path]:
    result = []
    for line in mountinfo.splitlines():
        sections = line.split(" - ", 1)
        fields = sections[0].split()
        if len(sections) != 2 or len(fields) < 6 or len(sections[1].split()) < 3:
            raise CacheCleanupError("malformed mountinfo; refusing cache cleanup")
        decoded = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
        mountpoint = Path(decoded)
        if not mountpoint.is_absolute() or ".." in mountpoint.parts:
            raise CacheCleanupError("invalid mountpoint; refusing cache cleanup")
        result.append(mountpoint)
    if not result:
        raise CacheCleanupError("empty mountinfo; refusing cache cleanup")
    return result


def _check_mounts(parent: Path, roots: Sequence[Path], mounts: Sequence[Path]) -> None:
    for mount in mounts:
        if mount == Path("/"):
            continue  # The container filesystem itself is necessarily mounted.
        if mount == parent or mount in parent.parents:
            raise CacheCleanupError(f"mounted cache ancestor: {mount}")
        for root in roots:
            target = root / "torch_compile_cache"
            if mount == root or mount == target or target in mount.parents:
                raise CacheCleanupError(f"mounted cache root/target/descendant: {mount}")


def clear_vllm_compile_caches(
    cache_parent: Path, mountinfo: str, environ: Mapping[str, str]
) -> int:
    """Validate the entire target set, delete compiled trees, return its size.

    Siblings such as HF data, dummy models and non-compilation vLLM data are
    never selected. Descendant symlinks are unlinked by fd-safe rmtree, not
    followed. An error is fatal; never report a partially cleared node as fresh.
    """
    _check_environment(environ)
    if not shutil.rmtree.avoids_symlink_attacks:
        raise CacheCleanupError("fd-safe rmtree unavailable; refusing cache cleanup")
    mounts = _mountpoints(mountinfo)
    parent_exists = _directory_exists(cache_parent)
    roots = []
    if parent_exists:
        # Shallow discovery only: no recursive walk of unrelated cache data.
        roots = sorted(path for path in cache_parent.iterdir() if _CACHE_NAME.fullmatch(path.name))
    _check_mounts(cache_parent, roots, mounts)
    targets = []
    for root in roots:
        _directory_exists(root)  # Refuse root symlinks even with an absent target.
        target = root / "torch_compile_cache"
        if _directory_exists(target):
            targets.append(target)

    # Nothing is removed until every candidate has passed validation.
    for target in targets:
        if not _directory_exists(target):
            raise CacheCleanupError(f"cache target changed during cleanup: {target}")
        shutil.rmtree(target)
        if os.path.lexists(target):
            raise CacheCleanupError(f"cache target remains after cleanup: {target}")
    return len(targets)


def main(argv=None) -> int:
    try:
        if argv is None:
            argv = sys.argv[1:]
        if argv:
            raise CacheCleanupError("this helper accepts no path or other arguments")
        mountinfo = Path("/proc/self/mountinfo").read_text()
        count = clear_vllm_compile_caches(CACHE_PARENT, mountinfo, os.environ)
        status = "deleted" if count else "already_absent"
        print(
            f"GRPO_VLLM_COMPILE_CACHE_CLEAR: node={socket.gethostname()} "
            f"status={status} targets={count}",
            flush=True,
        )
        return 0
    except (CacheCleanupError, OSError, RuntimeError) as error:
        print(f"GRPO_VLLM_COMPILE_CACHE_CLEAR_ERROR: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
