# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from generative_recommenders.dlrm_v4.train import m4_otf

logger = logging.getLogger(__name__)

ENV_ON = "DLRM_HSTU_FP8_M4"
ENV_TIMEOUT = "DLRM_HSTU_FP8_M4_FACTOR_TIMEOUT_S"
ENV_LOG_ALL = "DLRM_HSTU_FP8_M4_LOG_ALL"
ENV_LAG_MARGIN = "DLRM_HSTU_FP8_M4_LAG_MARGIN"
ENV_HELPER_TIMEOUT = "DLRM_HSTU_FP8_M4_HELPER_TIMEOUT_S"
ENV_HOST_LOCAL = "DLRM_HSTU_FP8_M4_HOST_LOCAL"


COST_B_MS = 38.157e-6
COST_C_MS = 13.509e-9
N_LIMIT = 4096
COST_QUANTUM_MS = 1e-6
WEIGHT_QUANTUM = 1 << 24
LAG_MARGIN_BATCHES = 32


MIN_LAG_MARGIN = 3


MAX_FACTOR_SPREAD = 4.0
DEFAULT_FACTOR_TIMEOUT_S = 600.0
DEFAULT_HELPER_TIMEOUT_S = 120.0
HELPER_START_TIMEOUT_S = 300.0
_HASH_MASK = (1 << 63) - 1


def switch_on() -> bool:
    return os.environ.get(ENV_ON, "0").strip() == "1"


def cost_lut(b_ms: float = COST_B_MS, c_ms: float = COST_C_MS, n_limit: int = N_LIMIT) -> np.ndarray:
    n = np.arange(n_limit, dtype=np.float64)
    lut = np.rint((b_ms * n + c_ms * n * (n + 1.0) * 0.5) / COST_QUANTUM_MS).astype(np.int64)
    if np.any(np.diff(lut[1:]) <= 0):
        raise ValueError("m4: cost LUT is not strictly increasing for n >= 1")
    return lut


def factors_valid(g, world_size: int) -> str:
    try:
        a = np.asarray(g, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return "not-numeric"
    if a.shape[0] != world_size:
        return f"len {a.shape[0]} != world_size {world_size}"
    if not np.all(np.isfinite(a)):
        return "non-finite"
    if np.any(a <= 0.0):
        return "non-positive"
    if float(a.max()) > MAX_FACTOR_SPREAD * float(a.min()):
        return f"spread > {MAX_FACTOR_SPREAD:g}x"
    return ""


def weights_from_factors(g: np.ndarray, world_size: int) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    if g.shape[0] != world_size or not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError(f"m4: invalid speed factors {g!r} for world_size={world_size}")
    g = g / g.mean()
    w = np.rint(WEIGHT_QUANTUM / g).astype(np.int64)
    if np.any(w <= 0) or int(w.max()) >= (1 << 31):
        raise ValueError(f"m4: weights out of range from factors {g!r}")
    return w


def _check_headroom(total_max: int, w_max: int) -> None:
    if int(total_max) * int(w_max) >= (1 << 62):
        raise ValueError(f"m4: target product {total_max} * {w_max} overflows the int64 headroom")


def partition_owner(n_blocks: np.ndarray, lut: np.ndarray, weights_q: np.ndarray, bs: int) -> np.ndarray:
    return partition_owner_load(n_blocks, lut, weights_q, bs)[0]


def partition_owner_load(n_blocks: np.ndarray, lut: np.ndarray, weights_q: np.ndarray, bs: int):
    n_blocks = np.asarray(n_blocks)
    if n_blocks.ndim != 2:
        raise ValueError("m4: n_blocks must be 2-D [nb, bs*W]")
    nb, G = n_blocks.shape
    W = int(weights_q.shape[0])
    if G != bs * W:
        raise ValueError(f"m4: block width {G} != bs*W = {bs}*{W}")
    owner = np.empty((nb, G), dtype=np.int16)
    if nb == 0:
        return owner, np.zeros((0, W), dtype=np.int64)
    nmin, nmax = int(n_blocks.min()), int(n_blocks.max())
    if nmin < 0 or nmax >= lut.shape[0]:
        raise ValueError(f"m4: N out of range [{nmin}, {nmax}]")
    costs = lut[n_blocks]

    key = (lut.shape[0] - 1) - n_blocks.astype(np.uint16)
    order = np.argsort(key, axis=1, kind="stable")
    sc = np.take_along_axis(costs, order, axis=1)
    total = costs.sum(axis=1)
    _check_headroom(int(total.max()), int(weights_q.max()))
    wsum = int(weights_q.sum())
    target = (total[:, None] * weights_q[None, :].astype(np.int64)) // wsum
    load = np.zeros((nb, W), dtype=np.int64)


    sc3 = sc.reshape(nb, W, bs)
    own3 = np.empty((nb, W, bs), dtype=np.int16)
    rows = np.arange(nb)[:, None]
    for k in range(bs):
        chunk = sc3[:, :, k]

        ranks = np.argsort(load - target, axis=1, kind="stable")
        own3[:, :, k] = ranks
        load[rows, ranks] += chunk
    np.put_along_axis(owner, order, own3.reshape(nb, G), axis=1)
    return owner, load


def offsets_from_ms(f, world_size: int) -> Optional[np.ndarray]:
    if f is None:
        return None
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    if f.shape[0] != world_size or not np.all(np.isfinite(f)):
        raise ValueError(f"m4: invalid offsets {f!r} for world_size={world_size}")
    return np.rint(np.maximum(f, 0.0) / COST_QUANTUM_MS).astype(np.int64)


def host_keys(hosts, world_size: int) -> Optional[np.ndarray]:
    if hosts is None:
        return None
    h = np.asarray(hosts, dtype=np.int64).reshape(-1)
    if h.shape[0] != world_size or np.any(h < 0) or np.any(h >= world_size):
        raise ValueError(f"m4: invalid host map {h!r} for world_size={world_size}")
    return h


def partition_owner_load_np(n_blocks: np.ndarray, lut: np.ndarray, weights_q: np.ndarray, bs: int,
                            offsets_q: Optional[np.ndarray] = None, hosts: Optional[np.ndarray] = None):


    n_blocks = np.asarray(n_blocks)
    if n_blocks.ndim != 2:
        raise ValueError("m4: n_blocks must be 2-D [nb, bs*W]")
    nb, G = n_blocks.shape
    W = int(weights_q.shape[0])
    if G != bs * W:
        raise ValueError(f"m4: block width {G} != bs*W = {bs}*{W}")
    owner = np.empty((nb, G), dtype=np.int16)
    if nb == 0:
        return owner, np.zeros((0, W), dtype=np.int64)
    nmin, nmax = int(n_blocks.min()), int(n_blocks.max())
    if nmin < 0 or nmax >= lut.shape[0]:
        raise ValueError(f"m4: N out of range [{nmin}, {nmax}]")
    wq = weights_q.astype(np.int64)
    costs = lut[n_blocks]
    n3 = n_blocks.reshape(nb, bs, W)
    rr_max = n3.max(axis=1).astype(np.int64)

    m_star = np.argmax(n3 == rr_max[:, None, :].astype(n3.dtype), axis=1)
    r_idx = np.arange(W)
    pre_pos = m_star * W + r_idx[None, :]

    hk = host_keys(hosts, W)
    if hk is None:
        key = np.arange(nb, dtype=np.int64)[:, None] * 4096 + rr_max
    else:
        nh = int(hk.max()) + 1
        key = (np.arange(nb, dtype=np.int64)[:, None] * nh + hk[None, :]) * 4096 + rr_max
    ukey, sub_id = np.unique(key.reshape(-1), return_inverse=True)
    sub_id = sub_id.reshape(nb, W)
    S = ukey.shape[0]
    Wg = np.bincount(sub_id.reshape(-1), minlength=S)

    order_r = np.argsort(sub_id.reshape(-1), kind="stable")
    seg_start = np.concatenate([[0], np.cumsum(Wg)[:-1]])
    slot_flat = np.empty(nb * W, dtype=np.int64)
    slot_flat[order_r] = np.arange(nb * W) - np.repeat(seg_start, Wg)
    slot = slot_flat.reshape(nb, W)
    rank_of_slot = np.full((S, W), -1, dtype=np.int64)
    rank_of_slot[sub_id.reshape(-1), slot_flat] = np.tile(r_idx, nb)

    pre_cost = np.take_along_axis(costs, pre_pos, axis=1)

    tot_int = np.zeros(S, dtype=np.int64)
    np.add.at(tot_int, sub_id.reshape(-1), costs.reshape(nb, bs, W).sum(axis=1).reshape(-1))
    _check_headroom(int(tot_int.max()), int(wq.max()))
    w_slot = np.zeros((S, W), dtype=np.int64)
    w_slot[sub_id.reshape(-1), slot_flat] = np.tile(wq, nb)
    wsum = w_slot.sum(axis=1)
    if offsets_q is None:
        target = (tot_int[:, None] * w_slot) // np.maximum(wsum, 1)[:, None]
    else:
        oq = np.asarray(offsets_q, dtype=np.int64).reshape(-1)
        if oq.shape[0] != W or np.any(oq < 0):
            raise ValueError("m4: offsets_q must be int64 [W] >= 0")
        f_slot = np.zeros((S, W), dtype=np.int64)
        f_slot[sub_id.reshape(-1), slot_flat] = np.tile(oq, nb)
        fw = (f_slot * w_slot) // WEIGHT_QUANTUM
        a = tot_int + fw.sum(axis=1)
        _check_headroom(int(a.max()), int(wq.max()))
        target = (a[:, None] * w_slot) // np.maximum(wsum, 1)[:, None] - fw
    load = np.zeros((S, W), dtype=np.int64)
    load[sub_id.reshape(-1), slot_flat] = pre_cost.reshape(-1)

    is_pre = np.zeros((nb, G), dtype=bool)
    np.put_along_axis(is_pre, pre_pos, True, axis=1)
    owner_rr_rank = np.tile(r_idx, bs)[None, :].repeat(nb, axis=0)
    sub_of_pos = np.take_along_axis(sub_id, owner_rr_rank, axis=1)
    rem_b, rem_p = np.nonzero(~is_pre)
    rem_sub = sub_of_pos[rem_b, rem_p]
    rem_n = n_blocks[rem_b, rem_p].astype(np.int64)
    srt = np.lexsort((rem_p, -rem_n, rem_sub))
    rem_b, rem_p, rem_sub = rem_b[srt], rem_p[srt], rem_sub[srt]
    R = bs - 1
    owner[rem_b, rem_p] = -1
    if R > 0:
        cnt = Wg * R
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        q = np.arange(rem_sub.shape[0]) - np.repeat(start, cnt)
        m_of = q // R
        k_of = q % R
        item_cost = lut[n_blocks[rem_b, rem_p]]
        slot_won = np.zeros(rem_sub.shape[0], dtype=np.int64)


        wclass = np.left_shift(1, np.ceil(np.log2(np.maximum(Wg, 1))).astype(np.int64))
        one = np.flatnonzero(wclass[rem_sub] == 1)
        np.add.at(load, (rem_sub[one], 0), item_cost[one])
        for wc in np.unique(wclass):
            wc = int(wc)
            if wc <= 1:
                continue
            subs = np.flatnonzero(wclass == wc)
            Sc = subs.shape[0]
            loc = np.full(S, -1, dtype=np.int64)
            loc[subs] = np.arange(Sc)
            sel = np.flatnonzero(loc[rem_sub] >= 0)
            ls, ms_, ks = loc[rem_sub[sel]], m_of[sel], k_of[sel]
            it = np.zeros((R, Sc, wc), dtype=np.int64)
            it[ks, ls, ms_] = item_cost[sel]
            ld = np.zeros((Sc, wc), dtype=np.int64)
            tg = np.full((Sc, wc), -(np.int64(1) << 61), dtype=np.int64)
            wmax = min(wc, W)
            ld[:, :wmax] = load[subs, :wmax]
            real = ~(rank_of_slot[subs, :wmax] < 0)
            tg[:, :wmax] = np.where(real, target[subs, :wmax], tg[:, :wmax])
            ld[:, :wmax] = np.where(real, ld[:, :wmax], 0)
            osl = np.empty((R, Sc, wc), dtype=np.int16)
            rows = np.arange(Sc)[:, None]
            for k in range(R):
                ordk = np.argsort(ld - tg, axis=1, kind="stable")
                osl[k] = ordk
                ld[rows, ordk] += it[k]
            load[subs, :wmax] = np.where(real, ld[:, :wmax], load[subs, :wmax])
            slot_won[sel] = osl[ks, ls, ms_]
        owner[rem_b, rem_p] = rank_of_slot[rem_sub, slot_won]
    np.put_along_axis(owner, pre_pos, np.tile(r_idx, nb).reshape(nb, W).astype(np.int16), axis=1)

    load_r = np.zeros((nb, W), dtype=np.int64)
    load_r[np.repeat(np.arange(nb), W), np.tile(r_idx, nb)] = load[sub_id.reshape(-1), slot_flat]
    return owner, load_r


def lag_k(j: int, nbs: List[int], depth: int, margin: int = LAG_MARGIN_BATCHES) -> int:
    if j < 2:
        return -1
    P = [0]
    for i in range(j):
        P.append(P[-1] + int(nbs[i]))
    s = j - 2
    while s - 1 >= 0 and (P[j] - P[s]) < 2 * depth:
        s -= 1
    best = -1
    for k in range(0, s):
        if P[s] - P[k + 1] >= depth + margin:
            best = k
        else:
            break
    return best


def lag_k_all(nbs: List[int], depth: int, margin: int = LAG_MARGIN_BATCHES) -> List[int]:
    import bisect

    P = [0]
    for x in nbs:
        P.append(P[-1] + int(x))
    out = []
    for j in range(len(nbs) + 1):
        if j < 2:
            out.append(-1)
            continue

        s = max(0, bisect.bisect_right(P, P[j] - 2 * depth, 0, j - 1) - 1)

        out.append(bisect.bisect_right(P, P[s] - depth - margin, 1, s + 1) - 2)
    return out


def predicted_nb(size: int, W: int, bs: int, nb_cap=None) -> int:
    if size < 0 or W <= 0 or bs <= 0:
        return 0
    nb = (int(size) // W) // bs
    return min(nb, max(0, int(nb_cap))) if nb_cap is not None else nb


def otf_nbs(nbs: List[int], sizes: List[int], W: int, bs: int, nb_cap=None) -> List[int]:
    full = [int(x) for x in nbs[:len(sizes)]]
    return full + [predicted_nb(sizes[j], W, bs, nb_cap) for j in range(len(full), len(sizes))]


def otf_need(full_nbs: List[int], depth: int, margin: int = LAG_MARGIN_BATCHES) -> List[int]:
    ks = lag_k_all(full_nbs[:-1], depth, margin) if full_nbs else []
    return [k + 1 if k >= 1 else j for j, k in enumerate(ks)]


def _h63(*parts: bytes) -> int:
    h = hashlib.sha1()
    for p in parts:
        h.update(p)
    return int.from_bytes(h.digest()[:8], "little") & _HASH_MASK


def compute_rr_costs(n_win, lut, b0, nb, W, bs, rank) -> np.ndarray:
    G = W * bs
    lo = b0 * G
    blocks = np.asarray(n_win[lo:lo + nb * G]).reshape(nb, G)
    return lut[blocks[:, rank::W]].sum(axis=1) * COST_QUANTUM_MS


def compute_m4(n_win, lut, j, ts, n_anchors, nb, W, bs, wq, rank, oq=None, hosts=None):
    G = W * bs
    blocks = np.asarray(n_win[0:nb * G]).reshape(nb, G)
    owner, load = partition_owner_load_np(blocks, lut, wq, bs, oq, hosts)
    pos = np.flatnonzero((owner == rank).reshape(-1)).astype(np.int64)
    if pos.shape[0] != nb * bs:
        raise RuntimeError(f"m4: rank {rank} got {pos.shape[0]} anchors, want {nb * bs}")

    h = _h63(b"m4", np.int64([j, ts, n_anchors, nb]).tobytes(), wq.tobytes(),
             (b"" if oq is None else np.asarray(oq, np.int64).tobytes()),
             (b"" if hosts is None else b"hl" + np.asarray(hosts, np.int64).tobytes()), owner.tobytes())
    return pos, load[:, rank] * COST_QUANTUM_MS, h


def window_array(fd: int) -> np.ndarray:
    import mmap

    size = os.fstat(fd).st_size
    if size % 2:
        raise ValueError(f"m4: window N buffer of {size} bytes is not int16")
    if size == 0:
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ), dtype=np.int16)


class LocalBackend:
    def __init__(self, lut: np.ndarray, W: int, rank: int) -> None:
        self._lut, self._W, self._r = lut, W, rank

    def m4(self, j, ts, n_anchors, nb, bs, wq, oq=None, hosts=None, fd=-1):
        return compute_m4(window_array(fd), self._lut, j, ts, n_anchors, nb, self._W, bs, wq, self._r, oq, hosts)

    def close(self, timeout_s: float = 5.0) -> None:
        pass


def _helper_loop(conn) -> None:
    from multiprocessing import reduction

    try:
        W, rank, b_ms, c_ms = conn.recv()
    except (EOFError, OSError):
        return
    try:
        lut = cost_lut(b_ms, c_ms)
        err = ""
    except BaseException as e:
        lut, err = None, f"{type(e).__name__}: {e}"
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        seq = msg[0] if isinstance(msg, tuple) and msg else -1
        kind = msg[1] if isinstance(msg, tuple) and len(msg) > 1 else None
        fd = -1
        if kind == "m4":
            try:
                fd = reduction.recv_handle(conn)
            except (EOFError, OSError, RuntimeError):
                return
        try:
            if err:
                raise RuntimeError(err)
            body = msg[2:]
            if kind == "ping":
                rep = (seq, "ok", os.getpid())
            elif kind == "m4":
                j, ts, n_anchors, nb, bs, wq_bytes, oq_bytes, hosts_bytes = body
                wq = np.frombuffer(wq_bytes, dtype=np.int64)
                oq = None if oq_bytes is None else np.frombuffer(oq_bytes, dtype=np.int64)
                hosts = None if hosts_bytes is None else np.frombuffer(hosts_bytes, dtype=np.int64)
                n_win = window_array(fd)
                os.close(fd)
                fd = -1
                rep = (seq, "ok", compute_m4(n_win, lut, j, ts, n_anchors, nb, W, bs, wq, rank, oq, hosts))
                del n_win
            else:
                raise ValueError(f"unknown request {kind!r}")
        except Exception as e:
            rep = (seq, "error", f"{type(e).__name__}: {e}")
        finally:
            if fd >= 0:
                os.close(fd)
        try:
            conn.send(rep)
        except (EOFError, OSError):
            return


def _helper_entry(fd: int, ppid: int = 0, pdeath: int = 0) -> None:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if pdeath:
        try:
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0)
        except Exception:
            pass
    if ppid and os.getppid() != ppid:
        return
    from multiprocessing.connection import Connection

    conn = Connection(fd)
    try:
        _helper_loop(conn)
    finally:
        try:
            conn.close()
        except OSError:
            pass


_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class HelperBackend:
    def __init__(self, W: int, rank: int, timeout_s: float,
                 b_ms: float = COST_B_MS, c_ms: float = COST_C_MS) -> None:
        import socket
        import subprocess
        import sys
        from multiprocessing.connection import Connection

        self._dead = ""
        self._seq = 0
        self._lock = threading.Lock()
        self._timeout = float(timeout_s)
        a, b = socket.socketpair()
        try:
            fd = b.fileno()
            code = (
                "import sys; sys.path.insert(0, %r); "
                "from generative_recommenders.dlrm_v4.train.m4_partition import _helper_entry; "
                "_helper_entry(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))" % _PKG_ROOT
            )
            pdeath = 1 if threading.current_thread() is threading.main_thread() else 0
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            self._p = subprocess.Popen(
                [sys.executable, "-B", "-c", code, str(fd), str(os.getpid()), str(pdeath)],
                pass_fds=(fd,), close_fds=True, stdin=subprocess.DEVNULL, env=env,
            )
        except BaseException:
            a.close()
            b.close()
            raise
        b.close()
        self._conn = Connection(a.detach())
        self.pid = int(self._p.pid)
        try:
            self._conn.send((int(W), int(rank), float(b_ms), float(c_ms)))
        except BaseException as e:
            self._poison(f"init send failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"m4 helper pid {self.pid} did not start (returncode={self._p.poll()}): "
                               f"{type(e).__name__}: {e}; kill switch {ENV_ON}=0") from e

    def _poison(self, why: str) -> None:
        if not self._dead:
            self._dead = why
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            self._p.kill()
        except Exception:
            pass
        try:
            self._p.wait(timeout=5.0)
        except Exception:
            pass

    def _call(self, kind: str, *body, timeout_s: Optional[float] = None, fd: Optional[int] = None):
        from multiprocessing import reduction

        tmo = self._timeout if timeout_s is None else float(timeout_s)
        with self._lock:
            if self._dead:
                raise RuntimeError(f"m4 helper pid {self.pid} is unusable ({self._dead}); "
                                   f"kill switch {ENV_ON}=0")
            self._seq += 1
            seq = self._seq
            try:
                self._conn.send((seq, kind) + tuple(body))
                if fd is not None:
                    reduction.send_handle(self._conn, int(fd), self.pid)
                if not self._conn.poll(tmo):
                    raise TimeoutError(f"no answer within {tmo:.0f}s")
                rep = self._conn.recv()
                if not (isinstance(rep, tuple) and len(rep) == 3 and rep[0] == seq):
                    raise RuntimeError(f"out-of-sequence reply (want seq {seq})")
            except BaseException as e:
                self._poison(f"{kind!r} failed: {type(e).__name__}: {e}")
                raise RuntimeError(
                    f"m4 helper pid {self.pid} failed during {kind!r} (returncode={self._p.poll()}): "
                    f"{type(e).__name__}: {e}; kill switch {ENV_ON}=0"
                ) from e
        if rep[1] != "ok":
            raise RuntimeError(f"m4 helper failed on {kind!r}: {rep[2]}")
        return rep[2]

    def ping(self):
        return self._call("ping", timeout_s=max(self._timeout, HELPER_START_TIMEOUT_S))

    def m4(self, j, ts, n_anchors, nb, bs, wq, oq=None, hosts=None, fd=-1):
        return self._call("m4", int(j), int(ts), int(n_anchors), int(nb), int(bs), wq.tobytes(),
                          None if oq is None else np.asarray(oq, np.int64).tobytes(),
                          None if hosts is None else np.asarray(hosts, np.int64).tobytes(), fd=int(fd))

    def close(self, timeout_s: float = 5.0) -> None:
        import subprocess

        got = self._lock.acquire(timeout=max(0.0, float(timeout_s)))
        try:
            if got and not self._dead:
                try:
                    self._conn.send(None)
                except Exception:
                    pass
            if got:
                try:
                    self._conn.close()
                except Exception:
                    pass
            if not self._dead:
                self._dead = "closed"
        finally:
            if got:
                self._lock.release()
        try:
            self._p.wait(timeout=float(timeout_s) if got else 0.0)
        except subprocess.TimeoutExpired:
            try:
                self._p.kill()
                self._p.wait(timeout=float(timeout_s))
            except Exception:
                pass
        except Exception:
            pass


class M4Dealer:
    def __init__(
        self,
        rank: int,
        world_size: int,
        backend,
        dataset=None,
        speed=None,
        hosts: Optional[np.ndarray] = None,
        lut: Optional[np.ndarray] = None,
        local_world_size: int = 0,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)

        self._hosts = host_keys(hosts, self.world_size)
        self._be = backend

        self._ds = dataset
        self._off: Optional[np.ndarray] = None
        self._tsi = None
        self._lut = cost_lut() if lut is None else lut
        self._lws = int(local_world_size) or self.world_size
        self._otf: Optional[m4_otf.OTFRuntime] = None
        self._otf_name: Optional[str] = None
        self._speed = speed
        self._nbs: List[int] = []
        self._depth = 0
        m = int(os.environ.get(ENV_LAG_MARGIN, LAG_MARGIN_BATCHES))
        if m < MIN_LAG_MARGIN:
            logger.warning("DLRMV4 m4: %s=%d below the pipeline pull-ahead bound; using %d",
                           ENV_LAG_MARGIN, m, MIN_LAG_MARGIN)
            m = MIN_LAG_MARGIN
        self._margin = m
        self._hash: Dict[int, int] = {}
        self._pending: Dict[int, Tuple[int, int, int, int]] = {}
        self._lock = threading.Lock()
        self._timeout = float(os.environ.get(ENV_TIMEOUT, DEFAULT_FACTOR_TIMEOUT_S))
        self._log_all = os.environ.get(ENV_LOG_ALL, "0") == "1"
        self._checked = 0
        self.counts = {"m4": 0, "rr": 0}
        self._reasons: Dict[str, int] = {}
        self.deferred_costs = 0
        self.dropped_costs = 0
        self._engaged = False
        self._last_vec = None
        self._bad_vec_logged = set()
        self._closed = False


        self._bs = 0
        self._nb_cap = None
        self._sched_nbs: Optional[List[int]] = None
        self._sched_pushed = 0
        self._speed_is_off = False


    def set_depth(self, depth: int) -> None:
        self._depth = max(0, int(depth))

    def bind_index(self, ts_offsets, ts_indices) -> None:
        self._off = np.asarray(ts_offsets, dtype=np.int64)
        self._tsi = ts_indices

    def activate(self, depth: int, ts_list) -> bool:
        global _ACTIVE
        self.set_depth(depth)
        if self._depth <= 0:
            if self._speed is not None:
                self._speed.set_enabled(False, "train loader prefetch depth 0")
            logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: prefetch depth 0", self.rank)
            self.close()
            return False
        rs = _run_start_wall()
        if rs is None:

            if self._speed is not None:
                self._speed.set_enabled(False, "m4otf gate closed: no run_start in this process")
            logger.warning("DLRMV4 m4otf gate closed on rank %d: no run_start in this process (resume or no MLPerf "
                           "logger): no data read, M4 off on every rank from window boundary 0 (round-robin)",
                           self.rank)
            self.close()
            return False
        if _ACTIVE:
            raise RuntimeError("m4: a second dealer was activated in this process")
        ds = self._ds
        if hasattr(ds, "_ensure_streaming_index"):
            ds._ensure_streaming_index()
        off = getattr(ds, "_ts_offsets", None)
        tsi = getattr(ds, "_ts_indices", None)
        if off is None or tsi is None:
            raise RuntimeError("DLRM_HSTU_FP8_M4=1 needs the offline ts index (DL_WINDOW_INDEX=1)")
        self.bind_index(off, tsi)
        _ACTIVE.append(self)
        if self._speed is not None:
            self._speed.register_boundary_callback(self.boundary_check)
        ts = [int(t) for t in ts_list]
        nw = self._off.shape[0] - 1
        sizes = [int(self._off[t + 1] - self._off[t]) if 0 <= t < nw else -1 for t in ts]
        self._otf = m4_otf.OTFRuntime.start(self.rank, ts, sizes, m4_otf.data_paths(ds), self._lws, rs,
                                            name=self._otf_name)
        logger.warning(
            "DLRMV4 armed OK: m4 speed-aware dealing rank=%d/%d depth=%d lag_margin=%d law=%s dealing=%s "
            "N=on-the-fly (m4otf builder=%s connected=%s, windows=%d); kill switch %s=0",
            self.rank, self.world_size, self._depth, self._margin, getattr(self, "law", None),
            "global" if self._hosts is None else "host-local(hosts=%d)" % (int(self._hosts.max()) + 1),
            "elected" if self._otf.elected else "client", "yes" if self._otf.connected() else "NO",
            len(ts), ENV_ON,
        )
        return True

    def close(self, timeout_s: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        otf = self._otf
        if otf is not None:
            try:
                otf.close(timeout_s)
            except Exception as e:
                logger.warning("[teardown] m4otf close failed on rank %d: %s: %s", self.rank, type(e).__name__, e)
        try:
            self._be.close(timeout_s)
        except Exception as e:
            logger.warning("[teardown] m4 helper close failed on rank %d: %s: %s", self.rank, type(e).__name__, e)

    def stream_closed(self, why: str = "train window stream closed") -> None:
        fn = getattr(self._speed, "shutdown", None)
        if fn is not None:
            try:
                fn(why)
            except Exception as e:
                logger.warning("DLRMV4 m4 rank=%d: m4_speed.shutdown failed: %s: %s", self.rank, type(e).__name__, e)
        otf = self._otf
        if otf is not None:
            try:
                otf.close()
            except Exception as e:
                logger.warning("DLRMV4 m4otf rank=%d: close failed: %s: %s", self.rank, type(e).__name__, e)

    def summary(self) -> None:
        with self._lock:
            c = dict(self.counts)
            r = dict(self._reasons)
        logger.warning("DLRMV4 m4 summary rank=%d: windows m4=%d rr=%d; rr reasons %s", self.rank, c["m4"], c["rr"], r)
        otf = self._otf
        if otf is not None:
            logger.warning("DLRMV4 m4otf summary rank=%d: windows built=%d ready_at_decision=%d not_ready=%d "
                           "deferred_costs=%d dropped_costs=%d rejected=%d builder=%s%s", self.rank, otf.received,
                           otf.book.ready, otf.book.not_ready, self.deferred_costs, self.dropped_costs, otf.rejected,
                           otf.role(), (" stopped=(%s)" % otf.stopped) if otf.stopped else "")


    def _window_ok(self, ts: int, gi: np.ndarray) -> str:
        if self._off is None:
            return "no-window-index"
        if ts < 0 or ts + 1 >= self._off.shape[0]:
            return "ts-out-of-index"
        lo, hi = int(self._off[ts]), int(self._off[ts + 1])
        if gi.shape[0] != hi - lo:
            return "indices-not-window-slice(len)"
        if self._tsi is not None and gi.shape[0] and not np.array_equal(gi, self._tsi[lo:hi]):
            return "indices-not-window-slice"
        return ""

    def _factors(self, k: int) -> Optional[np.ndarray]:
        if k < 1 or self._speed is None:
            return None
        return self._speed.factors_for(k, timeout_s=self._timeout)

    def _offsets(self, k: int) -> Optional[np.ndarray]:
        fn = getattr(self._speed, "offsets_for", None)
        if fn is None or k < 1:
            return None
        return fn(k, timeout_s=self._timeout)

    def _ready(self, j: int, boundary: int) -> bool:
        otf = self._otf
        return otf is not None and otf.book.decided_ready(j, boundary, self._timeout)

    def _speed_off(self) -> bool:
        if self._speed_is_off:
            return True
        fn = getattr(self._speed, "enabled", None)
        if fn is None:
            return False
        try:
            off = not fn()
        except Exception:
            return False
        self._speed_is_off = bool(off)
        return self._speed_is_off

    def _rr_costs(self, j: int, ts: int, b0: int, nb: int, bs: int) -> Optional[np.ndarray]:
        otf = self._otf
        fd = otf.dup_fd(j, ts) if otf is not None else None
        if fd is None:
            return None
        try:
            n_win = window_array(fd)
        finally:
            os.close(fd)
        return compute_rr_costs(n_win, self._lut, b0, nb, self.world_size, bs, self.rank)


    def deal(self, j: int, ts: int, global_indices, skip_samples: int, nb_cap, bs: int) -> Tuple[np.ndarray, int]:
        t0 = time.perf_counter()
        W, r = self.world_size, self.rank
        gi = np.asarray(global_indices)
        n = (gi.shape[0] // W) * W
        if skip_samples < 0 or skip_samples > n // W:
            raise ValueError(
                f"skip_samples={skip_samples} out of [0, {n // W}] for rank={r} world_size={W}"
            )
        nb = (n // W - skip_samples) // bs
        if nb_cap is not None:
            nb = min(nb, max(0, int(nb_cap)))
        with self._lock:
            if j != len(self._nbs):
                raise RuntimeError(f"m4: windows pushed out of order (j={j}, have {len(self._nbs)})")
            nbs = list(self._nbs)
            self._bs = int(bs)
            if j >= 1:
                self._nb_cap = nb_cap
        k = lag_k(j, nbs, self._depth, self._margin) if self._depth > 0 else -1
        g = None
        fd = None
        otf = "-"
        why = self._window_ok(ts, gi)
        if skip_samples:
            reason = "skip_samples"
        elif why:
            reason = why
        elif nb == 0:
            reason = "empty"
        elif k < 1:
            reason = f"no-factors(k={k})"
        else:
            g = self._factors(k)
            reason = "" if g is not None else f"factors-None(k={k})"
            if g is not None:
                bad = factors_valid(g, W)
                if bad:
                    if k not in self._bad_vec_logged:
                        self._bad_vec_logged.add(k)
                        logger.warning("DLRMV4 m4 rank=%d: factors[k=%d] unusable (%s): %r; window j=%d dealt "
                                       "round-robin", r, k, bad, g, j)
                    reason, g = f"bad-factors(k={k})", None
            if g is not None:

                if not self._ready(j, k + 1):
                    reason, g, otf = f"otf-not-ready(k={k})", None, "not-ready@%d" % (k + 1)
                else:
                    otf = "ready@%d" % (k + 1)
                    fd = self._otf.dup_fd(j, ts)
                    if fd is None:
                        if not self._otf.closed():
                            raise RuntimeError(f"m4otf: window j={j} was ready at boundary {k + 1} on every rank but "
                                               f"its N is not held on rank {r}; kill switch {ENV_ON}=0")
                        reason, g = "stream-closed", None
        G = bs * W
        defer = None
        if g is None:

            per_rank = np.array(gi[:n][r::W][skip_samples:], copy=True)
            costs_ms = None
            if not why and skip_samples % bs == 0 and nb > 0 and not self._speed_off():
                costs_ms = self._rr_costs(j, ts, skip_samples // bs, nb, bs)
                if costs_ms is None:
                    defer = (ts, skip_samples // bs, nb, bs)
            h = _h63(b"rr", np.int64([j, ts, n, skip_samples, nb]).tobytes())
            mode, vec = "rr", "-"
        else:
            try:
                wq = weights_from_factors(g, W)
                oq = offsets_from_ms(self._offsets(k), W)
                pos, costs_ms, h = self._be.m4(j, ts, gi.shape[0], nb, bs, wq, oq, self._hosts, fd)
            finally:
                os.close(fd)
            if pos.shape[0] != nb * bs:
                raise RuntimeError(f"m4: rank {r} got {pos.shape[0]} anchors, want {nb * bs}")
            if self._hosts is not None:


                rr = (pos % G) % W
                if not np.all(self._hosts[rr] == self._hosts[r]):
                    raise RuntimeError(f"m4: host-local deal of window j={j} gave rank {r} anchors of another host")
            per_rank = np.asarray(gi[: nb * G])[pos]
            mode, vec = "m4", "%016x" % _h63(wq.tobytes())
        with self._lock:
            self._nbs.append(int(nb))
            self._hash[j] = int(h)
            self.counts[mode] += 1
            if mode == "rr":
                key = reason.split("(")[0] or "none"
                self._reasons[key] = self._reasons.get(key, 0) + 1
        if mode == "m4" and not self._engaged:
            self._engaged = True
            logger.warning("DLRMV4 armed OK: m4 engaged rank=%d at j=%d ts=%d (k=%d)", r, j, ts, k)
        if mode == "m4" and r == 0 and vec != self._last_vec:
            self._last_vec = vec
            logger.warning("[m4] factors rank=0 k=%d vec=%s wq=%s g=%s", k, vec, wq.tolist(),
                           np.round(np.asarray(g, np.float64), 6).tolist())
        if self._speed is not None:


            self._speed.note_window_costs(
                j, np.asarray(costs_ms, dtype=np.float64) if costs_ms is not None else np.zeros(0, np.float64))
            if defer is not None:
                with self._lock:
                    self._pending[j] = defer
        if r == 0 or self._log_all:
            logger.warning(
                "[m4] rank=%d j=%d ts=%d k=%d mode=%s%s nb=%d vec=%s owner=%016x ms=%.1f otf=%s costs=%s",
                r, j, ts, k, mode, (" reason=" + reason) if reason else "", nb, vec, h,
                (time.perf_counter() - t0) * 1e3, otf,
                "helper" if mode == "m4" else ("now" if costs_ms is not None else ("deferred" if defer else "none")),
            )
        return per_rank, nb


    def window_hash(self, j: int) -> Optional[int]:
        with self._lock:
            return self._hash.get(j)

    def _gather(self, vals: List[int]) -> Optional[List[List[int]]]:
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
            return None
        dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        t = torch.tensor(vals, dtype=torch.int64, device=dev)
        out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(out, t)
        return [[int(v) for v in row] for row in torch.stack(out).cpu().tolist()]

    def boundary_check(self, i: int) -> None:
        if i == 0:
            return
        otf = self._otf
        w0 = w1 = 0
        if otf is not None:
            otf.drain(i)
            w0, w1 = otf.bits(i)
        h = self.window_hash(i)
        mine = -1 if h is None else int(h)
        rows = self._gather([mine, w0, w1])
        if rows is not None:
            vals = [x[0] for x in rows]
            if mine == -1 or any(v != vals[0] for v in vals):
                raise RuntimeError(
                    f"DLRMV4 m4 owner-hash MISMATCH at window j={i}: "
                    f"{['%016x' % v if v >= 0 else 'none' for v in vals]} (rank {self.rank}); "
                    f"kill switch DLRM_HSTU_FP8_M4=0"
                )
            self._checked += 1
            if self._checked == 1:
                logger.warning(
                    "DLRMV4 armed OK: m4 owner-hash check at window boundary j=%d: %d ranks agree (%016x)",
                    i, len(vals), vals[0],
                )
        else:
            rows = [[mine, w0, w1]]
        if otf is None:
            return
        a0 = a1 = -1
        for x in rows:
            a0 &= int(x[1])
            a1 &= int(x[2])
        otf.book.record(i, a0, a1)
        if self._speed_off():


            with self._lock:
                dropped = len(self._pending)
                self._pending.clear()
            self.dropped_costs += dropped
            otf.stop_builder(f"M4 is off for good from window boundary {i} (m4_speed)")
            return
        self._deferred(i)
        if otf.elected and otf.builder is not None and not otf.stopped:
            self._send_schedule(otf)
            with self._lock:
                done = sum(self._nbs[:i])
            otf.progress(i, done)

    def _send_schedule(self, otf) -> None:
        with self._lock:
            nbs = list(self._nbs)
            bs, cap = self._bs, self._nb_cap
        if bs <= 0 or self._depth <= 0 or otf.builder is None or otf.builder.dead:
            return
        basis = self._sched_nbs
        if basis is not None:
            if all(j < len(basis) and nbs[j] == basis[j] for j in range(self._sched_pushed, len(nbs))):
                self._sched_pushed = len(nbs)
                return
        full = otf_nbs(nbs, otf.sizes, self.world_size, bs, cap)
        if otf.schedule(otf_need(full, self._depth, self._margin)):
            self._sched_nbs = full
            self._sched_pushed = len(nbs)

    def _deferred(self, i: int) -> None:
        with self._lock:
            pend = sorted(self._pending.items())
        for j, (ts, b0, nb, bs) in pend:
            costs = self._rr_costs(j, ts, b0, nb, bs) if j >= i else None
            if costs is None and j > i:
                continue
            if costs is not None:
                if self._speed is not None:
                    self._speed.note_window_costs(j, np.asarray(costs, dtype=np.float64))
                self.deferred_costs += 1
            else:
                self.dropped_costs += 1
            with self._lock:
                self._pending.pop(j, None)


_DISABLED_REASON: List[str] = []
_ACTIVE: List["M4Dealer"] = []
_CREATED: List["M4Dealer"] = []


def decline_if_requested(rank: int, reason: str) -> None:
    if not switch_on():
        return
    try:
        from generative_recommenders.dlrm_v4.train import m4_speed as m4s
    except ImportError as e:
        raise RuntimeError(f"DLRM_HSTU_FP8_M4=1 but m4_speed is unavailable: {e}") from e
    _DISABLED_REASON.append(reason)
    m4s.set_enabled(False, reason)
    logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: %s", rank, reason)


def _run_start_wall() -> Optional[float]:
    from generative_recommenders.dlrm_v4.train import dl_prestart

    return dl_prestart.run_start_wall()


def maybe_create(dataset, rank: int, world_size: int, single_pool: bool) -> Optional[M4Dealer]:
    if not switch_on():
        return None


    for d in [d for d in _CREATED if d not in _ACTIVE]:
        logger.warning("DLRMV4 m4: closing the unused dealer of an earlier train loader (rank %d)", rank)
        d.close()
    _CREATED[:] = [d for d in _CREATED if d in _ACTIVE]
    try:
        from generative_recommenders.dlrm_v4.train import m4_speed as m4s
    except ImportError as e:
        raise RuntimeError(f"DLRM_HSTU_FP8_M4=1 but m4_speed is unavailable: {e}") from e
    reason = ""
    hosts = None
    nnodes = _nnodes()
    walk, walk_why = _walk_on()
    hl_mode = os.environ.get(ENV_HOST_LOCAL, "").strip().lower() or "auto"
    walk_global = False
    if not single_pool:
        reason = "DL_SINGLE_POOL=0 (double-buffered set_window path)"
    elif world_size <= 1:
        reason = "world_size<=1"
    elif hl_mode not in ("auto", "1", "0"):
        reason = f"{ENV_HOST_LOCAL}={hl_mode!r} is not auto/1/0 (fail closed)"
    elif nnodes > 1 and (hl_mode == "1" or (hl_mode == "auto" and walk)):
        hosts, reason = host_map(world_size, nnodes)
        if hosts is not None:
            reason = _layout_check(rank, hosts)
    elif nnodes > 1 and walk:
        walk_global = True
    if reason:
        _DISABLED_REASON.append(reason)
        m4s.set_enabled(False, reason)
        logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: %s", rank, reason)
        return None
    if hasattr(m4s, "enabled") and not m4s.enabled():
        logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: m4_speed disabled", rank)
        return None
    ds = dataset
    missing = [a for a in ("_ensure_streaming_index", "_cache_dir", "_positions_name", "_streaming_window_seconds")
               if not hasattr(ds, a)]
    if missing or not getattr(ds, "_use_offline_ts_indices", True):
        raise RuntimeError("DLRM_HSTU_FP8_M4=1 needs the streaming dataset with the offline ts index "
                           f"(DL_WINDOW_INDEX=1); missing {missing or ['_use_offline_ts_indices']}")


    differs = m4_otf.nrule_config_mismatch(ds)
    if differs:
        reason = (f"the dataset configuration differs from the m4otf N rule's ({differs}); N would not be the "
                  "model's sequence length")
        _DISABLED_REASON.append(reason)
        m4s.set_enabled(False, reason)
        logger.warning("DLRMV4 m4 speed-aware dealing DECLINED on rank %d: %s", rank, reason)
        return None
    t0 = time.time()


    b_ms = float(getattr(m4s, "COST_B_MS", getattr(m4s, "COST_B", COST_B_MS)))
    c_ms = float(getattr(m4s, "COST_C_MS", getattr(m4s, "COST_C", COST_C_MS)))
    lut = cost_lut(b_ms, c_ms)
    backend = HelperBackend(world_size, rank,
                            float(os.environ.get(ENV_HELPER_TIMEOUT, DEFAULT_HELPER_TIMEOUT_S)), b_ms=b_ms, c_ms=c_ms)
    try:
        backend.ping()
    except BaseException:
        backend.close()
        raise
    law_src = "m4_speed" if hasattr(m4s, "COST_B_MS") or hasattr(m4s, "COST_B") else "local default"
    lws = world_size // nnodes if world_size % nnodes == 0 else world_size
    dealer = M4Dealer(rank, world_size, backend, ds, speed=m4s, hosts=hosts, lut=lut, local_world_size=lws)
    dealer.law = (b_ms, c_ms, law_src)
    _CREATED.append(dealer)
    logger.warning(
        "DLRMV4 m4 dealer created rank=%d/%d (N on the fly after RUN_START (m4otf), law b=%.4g c=%.4g ms from %s, "
        "helper pid %d, ready %.1fs; dealing=%s (%s=%s, walk=%s%s, nnodes=%d); armed when the train stream starts)",
        rank, world_size, b_ms, c_ms, law_src, backend.pid, time.time() - t0,
        "global" if hosts is None else "host-local(hosts=%d)" % nnodes, ENV_HOST_LOCAL, hl_mode,
        "on" if walk else "off", (" (%s)" % walk_why) if walk_why else "", nnodes,
    )
    if walk_global:
        logger.warning(
            "DLRMV4 m4 dealing=global with the streaming page walk on across %d hosts (%s=0; examination "
            "setting): each host's walk advises its ranks' round-robin ranges, so anchors dealt to another host "
            "may read pages that host's walk did not advise; walk coverage approximate (rank %d)",
            nnodes, ENV_HOST_LOCAL, rank,
        )
    return dealer


def _nnodes() -> int:
    try:
        n = int(os.environ.get("NNODES", "").strip() or 1)
    except ValueError:
        n = 1
    return max(1, n)


def _walk_on() -> Tuple[bool, str]:
    try:
        from generative_recommenders.dlrm_v4.train.cold_prefetch import mode as _cp_mode
        from generative_recommenders.dlrm_v4.train.cold_prefetch import resolve_stream as _cp_resolve
    except ImportError:
        return False, "cold_prefetch not importable (no streaming walk)"
    if not _cp_mode():
        return False, "prefetcher off (cold_prefetch.mode()=0)"
    on, why = _cp_resolve()
    return bool(on), str(why or "")


def _layout_check(rank: int, hosts: np.ndarray) -> str:
    raw = os.environ.get("NODE_RANK", "").strip()
    try:
        nr = int(raw)
    except ValueError:
        return ""
    if 0 <= rank < hosts.shape[0] and int(hosts[rank]) != nr:
        return (f"host-local dealing: rank {rank} runs on NODE_RANK={nr} but the node-major host map puts it on "
                f"host {int(hosts[rank])} (rank layout is not node-major)")
    return ""


def host_map(world_size: int, nnodes: int) -> Tuple[Optional[np.ndarray], str]:
    if nnodes <= 1:
        return None, ""
    if world_size % nnodes:
        return None, f"host-local dealing: NNODES={nnodes} does not divide world_size={world_size}"
    gpn = world_size // nnodes
    raw = os.environ.get("GPUS_PER_NODE", "").strip()
    try:
        env_gpn = int(raw) if raw else 0
    except ValueError:
        env_gpn = 0
    if env_gpn > 0 and env_gpn != gpn:
        return None, (f"host-local dealing: GPUS_PER_NODE={env_gpn} != world_size/NNODES = "
                      f"{world_size}/{nnodes} (host map unknown)")
    return np.arange(world_size, dtype=np.int64) // gpn, ""
