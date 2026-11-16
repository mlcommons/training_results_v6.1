# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import array
import errno
import fcntl
import logging
import math
import os
import pickle
import re
import select
import socket
import struct
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


NFUNC_MODULE = "generative_recommenders.dlrm_v4.train.m4_nfunc"
NFUNC_NAME = "compute_window_n_incremental"

ENV_GO_BATCHES = "DLRM_HSTU_FP8_M4_OTF_GO_BATCHES"
ENV_J0 = "DLRM_HSTU_FP8_M4_OTF_J0"
ENV_LEAD = "DLRM_HSTU_FP8_M4_OTF_LEAD"
ENV_DUTY = "DLRM_HSTU_FP8_M4_OTF_DUTY"
DEFAULT_GO_BATCHES = 0
DEFAULT_J0 = 0
DEFAULT_LEAD = 16
DEFAULT_DUTY = 1.0
MIN_DUTY = 0.05


NAME_ENV = ("SLURM_JOB_ID", "SLURM_STEP_ID", "MASTER_PORT")
BUILDER_COMM = b"dlrmv4-m4otf"
BUILDER_MARK = "dlrmv4-m4otf-builder"

READY_SPAN = 128
MIN_EST_ANCHORS = 10000
CONNECT_RETRY_S = 2.0
LISTEN_BACKLOG = 64
N_MIN, N_MAX = 9, 4095
MAX_CTRL_MSG = 1 << 22
_HDR = struct.Struct("<qqqd")
_UCRED = struct.Struct("3i")
_M64 = (1 << 64) - 1
PR_SET_PDEATHSIG = 1
PR_SET_NAME = 15

F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 0x1)
F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 0x2)
F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 0x4)
F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 0x8)
REQUIRED_SEALS = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _env_duty() -> float:
    try:
        v = float(os.environ.get(ENV_DUTY, "").strip() or DEFAULT_DUTY)
    except ValueError:
        return DEFAULT_DUTY
    return min(1.0, max(MIN_DUTY, v)) if math.isfinite(v) else DEFAULT_DUTY


def _i64(x: int) -> int:
    x &= _M64
    return x - (1 << 64) if x >> 63 else x


def _now_ms() -> int:
    return int(time.time() * 1000)


def _stat_fields(pid: int) -> List[bytes]:
    with open(f"/proc/{int(pid)}/stat", "rb") as f:
        raw = f.read()
    return raw[raw.rindex(b")") + 2:].split()


def proc_start_time(pid: int) -> int:
    try:
        return int(_stat_fields(pid)[19])
    except (OSError, ValueError, IndexError):
        return 0


def proc_ppid(pid: int) -> int:
    try:
        return int(_stat_fields(pid)[1])
    except (OSError, ValueError, IndexError):
        return 0


def _job_tag() -> str:
    return ".".join(re.sub(r"[^0-9A-Za-z_]", "", os.environ.get(k, ""))[:20] for k in NAME_ENV)


def socket_name(ppid: Optional[int] = None) -> str:
    ppid = os.getppid() if ppid is None else int(ppid)
    return f"\0dlrmv4-m4otf-{ppid}-{proc_start_time(ppid)}-{_job_tag()}"


def _printable(name: str) -> str:
    return "@" + name[1:] if name.startswith("\0") else name


def peer_cred(sock: socket.socket) -> Tuple[int, int, int]:
    return _UCRED.unpack(sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size))


def check_peer(sock: socket.socket, parent: int) -> str:
    try:
        pid, uid, _ = peer_cred(sock)
    except OSError as e:
        return f"no peer credentials ({type(e).__name__}: {e})"
    if uid != os.getuid():
        return f"peer pid {pid} runs as uid {uid}"
    if parent:
        pp = proc_ppid(pid)
        if pp != int(parent):
            return f"peer pid {pid} is not a local rank (its parent {pp} != {int(parent)})"
    return ""


def set_proc_name(name: bytes) -> None:
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_NAME, ctypes.c_char_p(bytes(name)[:15]), 0, 0, 0)
    except Exception:
        pass


def _send_fd(sock: socket.socket, data: bytes, fd: int) -> int:
    return sock.sendmsg([data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [int(fd)]))],
                        socket.MSG_DONTWAIT | getattr(socket, "MSG_NOSIGNAL", 0))


def _recv_fd(sock: socket.socket, bufsize: int = 256) -> Tuple[bytes, List[int], int]:
    fds = array.array("i")
    msg, anc, flags, _ = sock.recvmsg(bufsize, socket.CMSG_SPACE(fds.itemsize),
                                      socket.MSG_DONTWAIT | getattr(socket, "MSG_CMSG_CLOEXEC", 0))
    for level, kind, data in anc:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds.frombytes(data[: len(data) - (len(data) % fds.itemsize)])
    return msg, list(fds), flags


def require_run_start(what: str) -> float:
    from generative_recommenders.dlrm_v4.train import dl_prestart

    rs = dl_prestart.run_start_wall()
    if rs is None:
        raise RuntimeError(f"m4otf: {what} before RUN_START (run-start gate closed); no data may be touched")
    return float(rs)


def data_paths(dataset) -> Dict[str, object]:
    cache = str(dataset._cache_dir)
    pname = str(dataset._positions_name)
    tag = pname[len("positions_"):-4] + f"_W{int(dataset._streaming_window_seconds)}"
    j = lambda f: os.path.join(cache, f)
    return {
        "positions": j(pname),
        "user_start": j("user_start.npy"),
        "flags": [j("flat_is_listen_plus.npy"), j("flat_is_like.npy"), j("flat_is_skip.npy")],
        "ts_indices": j(f"ts_indices_{tag}.npy"),
        "ts_offsets": j(f"ts_offsets_{tag}.npy"),
    }


NRULE_STRATEGY = "interleaved"
NRULE_HISTORY_LENGTH = 4086
NRULE_SCAN_WINDOW = 20000
NRULE_MAX_SEQ_LEN = 4096
NRULE_CANDIDATES = 1
NRULE_CONTEXTUAL = 8


def nrule_config_mismatch(dataset) -> str:
    why: List[str] = []
    miss = object()

    def get(name: str):
        v = getattr(dataset, name, miss)
        if v is miss:
            why.append(f"{name} missing")
        return v

    def want_int(name: str, want: int, where: str) -> None:
        v = get(name)
        if v is not miss and (isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) != want):
            why.append(f"{name}={v!r} (want {want}{where})")

    v = get("_history_strategy")
    if v is not miss and v != NRULE_STRATEGY:
        why.append(f"_history_strategy={v!r} (want {NRULE_STRATEGY!r}, $HISTORY_STRATEGY)")
    want_int("_history_length", NRULE_HISTORY_LENGTH, ", $HISTORY_LENGTH")
    want_int("_scan_window", NRULE_SCAN_WINDOW, ", get_dataset scan_window")
    want_int("_max_seq_len", NRULE_MAX_SEQ_LEN, ", $MAX_SEQ_LEN")
    v = get("_is_inference")
    if v is not miss and (not isinstance(v, (bool, np.bool_)) or bool(v)):
        why.append(f"_is_inference={v!r} (want False)")
    want_int("_max_num_candidates", NRULE_CANDIDATES, ", HSTU config")
    v = get("_contextual_feature_to_max_length")
    if v is not miss:
        try:
            lens = [int(x) for x in dict(v or {}).values()]
        except (TypeError, ValueError):
            lens = None
        if lens is None or len(lens) != NRULE_CONTEXTUAL or any(x != 1 for x in lens):
            got = ("not a feature -> length map" if lens is None
                   else f"{len(lens)} features, lengths {sorted(set(lens))}")
            why.append(f"_contextual_feature_to_max_length: {got} (want {NRULE_CONTEXTUAL} of length 1, HSTU config)")
    return "; ".join(why)


class Readiness:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._snap: Dict[int, int] = {}
        self._closed = False
        self.ready = 0
        self.not_ready = 0

    def record(self, i: int, w0: int, w1: int) -> None:
        bits = (int(w0) & _M64) | ((int(w1) & _M64) << 64)
        with self._cond:
            self._snap[int(i)] = bits
            for old in [b for b in self._snap if b < int(i) - 4 * READY_SPAN]:
                del self._snap[old]
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def decided_ready(self, j: int, boundary: int, timeout_s: float) -> bool:
        j, b = int(j), int(boundary)
        deadline = time.monotonic() + float(timeout_s)
        with self._cond:
            while b not in self._snap:
                if self._closed:
                    return False
                if threading.current_thread() is threading.main_thread():
                    raise RuntimeError(
                        f"m4otf: readiness of boundary {b} asked on the main thread before it was recorded "
                        "(the main thread records it): would deadlock")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        f"m4otf: readiness of window boundary {b} not recorded within {float(timeout_s):.0f} s "
                        f"(window j={j}); kill switch DLRM_HSTU_FP8_M4=0")
                self._cond.wait(remaining)
            d = j - b
            ok = 0 <= d < READY_SPAN and bool((self._snap[b] >> d) & 1)
            if ok:
                self.ready += 1
            else:
                self.not_ready += 1
            return ok


def _connect(name: str, budget_s: float = CONNECT_RETRY_S,
             parent: Optional[int] = None) -> Tuple[Optional[socket.socket], str]:


    parent = os.getppid() if parent is None else int(parent)
    t_end = time.monotonic() + float(budget_s)
    delay = 0.002
    while True:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            s.settimeout(1.0)
            s.connect(name)
        except OSError as e:
            s.close()
            if e.errno not in (errno.ECONNREFUSED, errno.ENOENT, errno.EAGAIN) and not isinstance(e, TimeoutError):
                return None, f"{type(e).__name__}: {e}"
            if time.monotonic() + delay > t_end:
                return None, f"refused for {float(budget_s):.1f} s ({type(e).__name__}: {e})"
            time.sleep(delay)
            delay = min(2 * delay, 0.1)
            continue
        why = check_peer(s, parent)
        if why:
            s.close()
            return None, f"the socket's listener is not a local rank: {why}"
        s.setblocking(False)
        return s, ""


class _BuilderHandle:
    def __init__(self, lsock: socket.socket, rank: int, local_world_size: int = 1,
                 parent: Optional[int] = None) -> None:
        import subprocess

        self.rank = int(rank)
        self.dead = ""
        self.go_sent = False
        self.dropped = 0
        self._exit_logged = False
        self._closed = False
        parent = os.getppid() if parent is None else int(parent)
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            code = (
                "import sys; sys.path.insert(0, %r); "
                "from generative_recommenders.dlrm_v4.train.m4_otf import _builder_entry; "
                "_builder_entry(*[int(x) for x in sys.argv[1:7]])" % _PKG_ROOT
            )
            pdeath = 1 if threading.current_thread() is threading.main_thread() else 0
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            self._p = subprocess.Popen(
                [sys.executable, "-B", "-c", code, str(b.fileno()), str(lsock.fileno()), str(os.getpid()),
                 str(pdeath), str(parent), str(max(1, int(local_world_size))), BUILDER_MARK],
                pass_fds=(b.fileno(), lsock.fileno()), close_fds=True, stdin=subprocess.DEVNULL, env=env,
            )
        except BaseException:
            a.close()
            raise
        finally:
            b.close()
        self._ctrl = a
        self.pid = int(self._p.pid)

    def _send(self, msg, block: bool) -> bool:
        if self.dead:
            return False
        data = pickle.dumps(msg, protocol=4)
        try:
            if block:
                self._ctrl.settimeout(5.0)
                self._ctrl.send(data)
                self._ctrl.setblocking(False)
            else:
                self._ctrl.send(data, socket.MSG_DONTWAIT)
            return True
        except (BlockingIOError, InterruptedError):
            self.dropped += 1
            return False
        except OSError as e:
            self.dead = f"{type(e).__name__}: {e}"
            return False

    def release(self, paths, ts_list: Sequence[int], plan: Dict[str, object], local_world_size: int) -> bool:
        require_run_start("m4otf release of the data paths")
        return self._send(("release", paths, [int(t) for t in ts_list], dict(plan), int(local_world_size)),
                          block=True)

    def schedule(self, need: Sequence[int]) -> bool:
        if self._closed:
            return False
        return self._send(("sched", [int(x) for x in need]), block=False)

    def progress(self, i: int, batches: int, go_batches: int) -> None:
        if self._closed:
            return
        self._send(("progress", int(i), int(batches), time.time()), block=False)
        if not self.go_sent and int(batches) >= int(go_batches):
            self.go_sent = self._send(("go",), block=False)
        rc = self._p.poll()
        if rc is not None and not self._exit_logged:
            self._exit_logged = True
            logger.warning("DLRMV4 m4otf builder failed: builder pid %d exited with code %s before the stream "
                           "ended (rank %d); unheld windows deal round-robin", self.pid, rc, self.rank)

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.dead = self.dead or "stopped"
        try:
            self._ctrl.send(pickle.dumps(None, protocol=4), socket.MSG_DONTWAIT)
        except Exception:
            pass
        try:
            self._ctrl.close()
        except Exception:
            pass

    def close(self, timeout_s: float = 5.0) -> None:
        import subprocess

        self._closed = True
        try:
            self._ctrl.send(pickle.dumps(None, protocol=4), socket.MSG_DONTWAIT)
        except Exception:
            pass
        try:
            self._ctrl.close()
        except Exception:
            pass
        self.dead = self.dead or "closed"
        try:
            self._p.wait(timeout=min(1.0, max(0.0, float(timeout_s))))
        except subprocess.TimeoutExpired:
            try:
                self._p.kill()
                self._p.wait(timeout=max(0.1, float(timeout_s)))
            except Exception:
                pass
        except Exception:
            pass


class OTFRuntime:
    def __init__(self, rank: int, ts_list: Sequence[int], sizes: Sequence[int]) -> None:
        self.rank = int(rank)
        self.ts_list = [int(t) for t in ts_list]
        self.sizes = [int(s) for s in sizes]
        self.book = Readiness()
        self.builder: Optional[_BuilderHandle] = None
        self.elected = False
        self.stopped = ""
        self.name = ""
        self.go_batches = max(0, _env_int(ENV_GO_BATCHES, DEFAULT_GO_BATCHES))
        self._lock = threading.Lock()
        self._fds: Dict[int, int] = {}
        self._sock: Optional[socket.socket] = None
        self._lsock: Optional[socket.socket] = None
        self._eof = False
        self._closed = False
        self.received = 0
        self.rejected = 0
        self._warned = set()


    @classmethod
    def start(cls, rank: int, ts_list: Sequence[int], sizes: Sequence[int], paths, local_world_size: int,
              run_start_wall: float, name: Optional[str] = None) -> "OTFRuntime":


        rt = cls(rank, ts_list, sizes)
        rt.name = socket_name() if name is None else name
        shown = _printable(rt.name)
        lsock = None
        try:
            lsock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            lsock.bind(rt.name)
            lsock.listen(LISTEN_BACKLOG)
        except OSError as e:
            if lsock is not None:
                lsock.close()
            lsock = None
            if e.errno != errno.EADDRINUSE:
                logger.warning("DLRMV4 m4otf rank=%d: election socket %s failed (%s: %s)", rank, shown,
                               type(e).__name__, e)
        if lsock is not None:

            rt._lsock = lsock
            rt.elected = True
            try:
                rt.builder = _BuilderHandle(lsock, rank, local_world_size)
                logger.warning("DLRMV4 m4otf builder elected rank=%d (local clients expected %d): builder pid %d, "
                               "socket %s", rank, int(local_world_size), rt.builder.pid, shown)
            except BaseException as e:
                logger.warning("DLRMV4 m4otf builder failed to start on rank %d: %s: %s", rank, type(e).__name__, e)
        sock, why = _connect(rt.name)
        rt._sock = sock
        if sock is None:
            logger.warning("DLRMV4 m4otf rank=%d: no connection to the node builder %s (%s): this rank reports "
                           "no window ready (round-robin on every rank through the readiness AND)", rank, shown, why)
        elif not rt.elected:
            logger.warning("DLRMV4 m4otf rank=%d: client of the node builder %s (elected on another local rank)",
                           rank, shown)
        if rt.builder is not None:
            plan = {"j0": max(0, _env_int(ENV_J0, DEFAULT_J0)), "lead": max(0, _env_int(ENV_LEAD, DEFAULT_LEAD)),
                    "duty": _env_duty(), "nfunc": NFUNC_MODULE}
            if rt.builder.release(paths, rt.ts_list, plan, local_world_size):
                logger.warning("DLRMV4 armed OK: m4otf release after run_start (+%.3fs) rank=%d builder pid %d "
                               "(j0=%d lead=%d duty=%g go_batches=%d windows=%d) t=%d run_start_ms=%d",
                               time.time() - float(run_start_wall), rank, rt.builder.pid, plan["j0"], plan["lead"],
                               plan["duty"], rt.go_batches, len(rt.ts_list), _now_ms(),
                               int(float(run_start_wall) * 1000))
            else:
                logger.warning("DLRMV4 m4otf builder failed: release not delivered to builder pid %d (%s) on rank "
                               "%d", rt.builder.pid, rt.builder.dead or "send failed", rank)
        return rt


    def _warn_once(self, key: str, msg: str, *args) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg, *args)

    def hold(self, j: int, fd: int) -> bool:
        why = ""
        if self._closed:
            why = "-"
        elif not (0 <= j < len(self.ts_list)) or self.sizes[j] < 0:
            why = f"unknown window j={j}"
        else:
            try:
                st = os.fstat(fd)
                seals = fcntl.fcntl(fd, F_GET_SEALS)
                if st.st_size != 2 * self.sizes[j]:
                    why = f"size {st.st_size} != 2 * {self.sizes[j]} anchors"
                elif (seals & REQUIRED_SEALS) != REQUIRED_SEALS:
                    why = f"not sealed (seals=0x{seals:x})"
            except OSError as e:
                why = f"not a sealed memfd ({type(e).__name__}: {e})"
        if not why:
            with self._lock:
                if not self._closed and j not in self._fds:
                    self._fds[j] = fd
                    self.received += 1
                    return True
            why = "-"
        if why != "-":
            self.rejected += 1
            self._warn_once("reject", "DLRMV4 m4otf rank=%d: rejected a published window (%s); that window "
                            "deals round-robin", self.rank, why)
        try:
            os.close(fd)
        except OSError:
            pass
        return False

    def drain(self, i: int) -> None:
        self.release_below(i)
        sock = self._sock
        if sock is None or self._eof:
            return
        trunc = getattr(socket, "MSG_TRUNC", 0) | getattr(socket, "MSG_CTRUNC", 0)
        while True:
            try:
                msg, fds, flags = _recv_fd(sock)
            except (BlockingIOError, InterruptedError):
                return
            except OSError as e:
                self._eof = True
                self._warn_once("recv", "DLRMV4 m4otf rank=%d: builder connection failed (%s: %s); later windows "
                                "deal round-robin", self.rank, type(e).__name__, e)
                return
            if not msg and not fds:
                self._eof = True
                return
            if len(msg) != _HDR.size or len(fds) != 1 or (flags & trunc):
                self.rejected += 1
                self._warn_once("malformed", "DLRMV4 m4otf rank=%d: malformed builder message (%d bytes, %d fds)",
                                self.rank, len(msg), len(fds))
                for fd in fds:
                    os.close(fd)
                continue
            ts, j, n, _ms = _HDR.unpack(msg)
            if 0 <= j < len(self.ts_list) and self.ts_list[j] == ts and self.sizes[j] == n:
                if j < i:
                    os.close(fds[0])
                else:
                    self.hold(j, fds[0])
            else:
                self.rejected += 1
                self._warn_once("mismatch", "DLRMV4 m4otf rank=%d: builder window (j=%d ts=%d anchors=%d) does not "
                                "match this rank's window index; dealt round-robin", self.rank, j, ts, n)
                os.close(fds[0])

    def release_below(self, i: int) -> None:
        with self._lock:
            old = [j for j in self._fds if j < i]
            fds = [self._fds.pop(j) for j in old]
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass

    def bits(self, i: int) -> Tuple[int, int]:
        with self._lock:
            held = [j - i for j in self._fds if i <= j < i + READY_SPAN]
        b = 0
        for d in held:
            b |= 1 << d
        return _i64(b), _i64(b >> 64)

    def holds(self, j: int) -> bool:
        with self._lock:
            return j in self._fds

    def dup_fd(self, j: int, ts: Optional[int] = None) -> Optional[int]:
        j = int(j)
        if ts is not None and not (0 <= j < len(self.ts_list) and self.ts_list[j] == int(ts)):
            return None
        with self._lock:
            fd = self._fds.get(j)
            return None if fd is None or self._closed else os.dup(fd)

    def closed(self) -> bool:
        return self._closed

    def schedule(self, need: Sequence[int]) -> bool:
        if self.builder is None or self._closed or self.stopped:
            return False
        return self.builder.schedule(need)

    def progress(self, i: int, batches: int) -> None:
        if self.builder is not None and not self._closed and not self.stopped:
            self.builder.progress(i, batches, self.go_batches)

    def stop_builder(self, why: str) -> None:
        if self.stopped:
            return
        self.stopped = why or "stopped"
        if self.builder is not None and not self._closed:
            logger.warning("DLRMV4 m4otf builder stopped rank=%d (builder pid %d): %s; no later window can use N",
                           self.rank, self.builder.pid, self.stopped)
            try:
                self.builder.stop()
            except Exception:
                pass

    def connected(self) -> bool:
        return self._sock is not None

    def close(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            fds = list(self._fds.values())
            self._fds.clear()
        self.book.close()
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self.builder is not None:
            try:
                self.builder.close(timeout_s)
            except Exception:
                pass
        if self._lsock is not None:
            try:
                self._lsock.close()
            except OSError:
                pass

    def role(self) -> str:
        return "elected" if self.elected else ("client" if self._sock is not None else "unconnected")

    def summary(self) -> str:
        return (f"windows built={self.received} ready_at_decision={self.book.ready} "
                f"not_ready={self.book.not_ready} rejected={self.rejected} builder={self.role()}"
                + (f" stopped=({self.stopped})" if self.stopped else ""))


def pick_window(progress: int, cursor: int, built: Set[int], need: Sequence[int], lead: int, j0: int,
                est_s: Optional[Callable[[int], float]] = None, boundary_s: Optional[float] = None,
                elapsed_s: float = 0.0) -> Tuple[Optional[int], int, str]:


    q, lead = int(progress), int(lead)
    n = len(need)
    lo = max(q + 1, int(j0))

    def deadline(j: int) -> Tuple[int, int]:
        d = int(need[j])
        return (d, 0) if q < d < j else (j, 1)

    hi = min(n - 1, q + lead)
    for j in range(hi + 1, n):
        d = int(need[j])
        if d < j:
            if d > q + lead:
                break
            if d > q:
                hi = j
    c = max(int(cursor), lo)
    while c <= hi and c in built:
        c += 1
    if c > hi:
        for j in range(lo, hi + 1):
            if j not in built:
                return j, c, "backfill"
        return None, c, "idle"
    if est_s is None or boundary_s is None or boundary_s <= 0.0:
        return c, c, "asc"

    def left_s(j: int) -> float:
        return (deadline(j)[0] - q) * float(boundary_s) - float(elapsed_s)

    cands = sorted((j for j in range(c, hi + 1) if j not in built), key=lambda x: (deadline(x), x))
    u = next((j for j in cands if est_s(j) <= left_s(j)), None)
    if u is not None and u != c and sum(est_s(j) for j in range(c, u + 1) if j not in built) > left_s(u):
        return u, u, "jump"
    return c, c, "asc"


def _say(msg: str) -> None:
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _ioprio_idle() -> None:
    try:
        import ctypes
        import platform

        nr = {"x86_64": 251, "aarch64": 30}.get(platform.machine())
        if nr is not None:
            ctypes.CDLL(None, use_errno=True).syscall(nr, 1, 0, 3 << 13)
    except Exception:
        pass


def _builder_entry(ctrl_fd: int, listen_fd: int, ppid: int = 0, pdeath: int = 0, parent: int = 0,
                   lws: int = 0) -> None:


    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if pdeath:
        try:
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
        except Exception:
            pass
    if ppid and os.getppid() != ppid:
        return
    set_proc_name(BUILDER_COMM)
    try:
        os.nice(10)
    except OSError:
        pass
    _ioprio_idle()
    ctrl = socket.socket(fileno=ctrl_fd)
    lsock = socket.socket(fileno=listen_fd)
    b = _Builder(ctrl, lsock, ppid if not pdeath else 0, parent=parent, max_clients=lws)
    _say(f"[m4otf] node={b.host} builder started pid={os.getpid()} ppid={os.getppid()} (no data path until "
         f"release and go) t={_now_ms()}")
    try:
        b.run()
    except BaseException as e:
        _say(f"[m4otf] node={b.host} m4otf builder failed: {type(e).__name__}: {e} t={_now_ms()}")
    finally:
        b.shutdown()


class _Builder:
    def __init__(self, ctrl: socket.socket, lsock: socket.socket, watch_ppid: int = 0, parent: int = 0,
                 max_clients: int = 0) -> None:
        self.ctrl, self.lsock, self.watch_ppid = ctrl, lsock, int(watch_ppid)
        self.parent = int(parent)
        self.max_clients = int(max_clients)
        self.lsock.setblocking(False)


        self._poller = select.poll()
        self._poller.register(self.ctrl, select.POLLIN)
        self._poller.register(self.lsock, select.POLLIN)
        self.host = socket.gethostname()
        self.clients: List[List[object]] = []
        self.accepted = 0
        self.refused = 0
        self.pub: Dict[int, List[object]] = {}
        self.seq = 0
        self.released = False
        self.go = False
        self.progress = -1
        self.j0 = DEFAULT_J0
        self.lead = DEFAULT_LEAD
        self.duty = DEFAULT_DUTY
        self.rest_until = 0.0
        self.lws = 1
        self.ts_list: List[int] = []
        self.need: List[int] = []
        self.built: Set[int] = set()
        self.cursor = 0
        self.n_built = 0
        self.blocked = False
        self.paths: Optional[Dict[str, object]] = None
        self.nfunc = NFUNC_MODULE
        self.state = None
        self.fn = None
        self.off: Optional[np.ndarray] = None
        self.sizes: List[int] = []
        self.user_start: Optional[np.ndarray] = None
        self.first_memfd = False


        self.t_prog: Optional[float] = None
        self.boundary_s: Optional[float] = None
        self.s_per_anchor: Optional[float] = None


    def _read_ctrl(self) -> bool:
        while True:
            try:
                data = self.ctrl.recv(MAX_CTRL_MSG, socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                return True
            if not data:
                return False
            msg = pickle.loads(data)
            if msg is None:
                return False
            kind = msg[0]
            if kind == "release" and not self.released:
                self._release(*msg[1:])
            elif kind == "sched":
                self.need = [int(x) for x in msg[1]]
            elif kind == "progress":
                self._progress(*msg[1:])
            elif kind == "go" and not self.go:
                self.go = True
                _say(f"[m4otf] node={self.host} go progress={self.progress} t={_now_ms()}")

    def _progress(self, i, batches=0, wall=None) -> None:
        i = int(i)
        if i <= self.progress:
            return
        if wall is not None:
            wall = float(wall)
            if self.t_prog is not None and self.progress >= 0 and wall > self.t_prog:
                per = (wall - self.t_prog) / float(i - self.progress)
                self.boundary_s = per if self.boundary_s is None else 0.5 * self.boundary_s + 0.5 * per
            self.t_prog = wall
        self.progress = i
        self.built = {j for j in self.built if j > i}

    def _release(self, paths, ts_list, plan, lws) -> None:
        self.ts_list = [int(t) for t in ts_list]
        self.lws = max(1, int(lws))
        self.j0 = max(0, int(plan.get("j0", DEFAULT_J0)))
        self.cursor = self.j0
        self.lead = max(0, int(plan.get("lead", DEFAULT_LEAD)))
        duty = float(plan.get("duty", DEFAULT_DUTY))
        self.duty = min(1.0, max(MIN_DUTY, duty)) if math.isfinite(duty) else DEFAULT_DUTY
        self.nfunc = str(plan.get("nfunc", NFUNC_MODULE))
        self.paths = {"positions": str(paths["positions"]), "user_start": str(paths["user_start"]),
                      "flags": [str(p) for p in paths["flags"]], "ts_indices": str(paths["ts_indices"]),
                      "ts_offsets": str(paths["ts_offsets"])}
        self.released = True
        _say(f"[m4otf] node={self.host} release received windows={len(self.ts_list)} lws={self.lws} j0={self.j0} "
             f"lead={self.lead} duty={self.duty:g} t={_now_ms()}")

    def _open_data(self) -> None:
        import importlib

        mod = importlib.import_module(self.nfunc)
        fn = getattr(mod, NFUNC_NAME)
        p = self.paths
        m = lambda f: np.load(f, mmap_mode="r")
        off = np.array(m(p["ts_offsets"]), dtype=np.int64)
        tsi = m(p["ts_indices"])
        if off.ndim != 1 or off.shape[0] < 2 or int(off[-1]) != int(tsi.shape[0]):
            raise ValueError("ts_offsets does not index ts_indices")
        positions = m(p["positions"])
        flags = []
        for f in p["flags"]:
            a = np.asarray(m(f))
            if a.ndim != 1 or a.dtype.itemsize != 1 or a.dtype.kind not in "biu":
                raise ValueError(f"pool flags must be 1-D bytes, got {a.dtype} ndim={a.ndim}")
            flags.append(a.view(np.uint8))
        n_events = int(flags[0].shape[0])
        if any(int(f.shape[0]) != n_events for f in flags):
            raise ValueError("pool flag columns differ in length")
        us = np.array(m(p["user_start"]), dtype=np.int64)
        us = us[us >= 0]
        if us.size == 0 or int(us[0]) != 0 or np.any(np.diff(us) <= 0) or int(us[-1]) >= n_events:
            raise ValueError("user_start is not an ascending tiling of the event range")
        self.off, self.tsi, self.positions, self.flags = off, tsi, positions, tuple(flags)
        self.user_start = np.append(us, np.int64(n_events)).astype(np.int64)
        self.n_users = int(us.size)
        nw = off.shape[0] - 1
        self.sizes = [int(off[t + 1] - off[t]) if 0 <= t < nw else 0 for t in self.ts_list]
        self.fn = fn
        _say(f"[m4otf] node={self.host} data mapped (7 files) t={_now_ms()}")


    def _need(self, j: int) -> int:
        return self.need[j] if j < len(self.need) and self.need[j] >= 0 else j

    def _est_s(self, j: int) -> float:
        n = self.sizes[j] if 0 <= j < len(self.sizes) else 0
        return float(self.s_per_anchor) * max(n, MIN_EST_ANCHORS)

    def _pick(self) -> Tuple[Optional[int], int, str]:
        if not (self.released and self.go) or time.monotonic() < self.rest_until:
            return None, self.cursor, "idle"
        need = [self._need(j) for j in range(len(self.ts_list))]
        est = self._est_s if (self.s_per_anchor is not None and self.sizes) else None
        elapsed = max(0.0, time.time() - self.t_prog) if self.t_prog is not None else 0.0
        return pick_window(self.progress, self.cursor, self.built, need, self.lead, self.j0, est, self.boundary_s,
                           elapsed)


    def _build(self, j: int, cursor: int, how: str) -> None:
        self.built.add(j)
        if how != "backfill":
            self.cursor = j + 1
        if self.fn is None:
            self._open_data()
        ts = self.ts_list[j]
        if not (0 <= ts < self.off.shape[0] - 1):
            return
        t0 = time.perf_counter()
        lo, hi = int(self.off[ts]), int(self.off[ts + 1])
        anchors = np.asarray(self.tsi[lo:hi], dtype=np.int64)
        p = np.asarray(self.positions[anchors], dtype=np.int64)
        n_win, self.state = self.fn(p, self.user_start, self.flags, self.state, user_lo=0, user_hi=self.n_users)
        n = hi - lo
        if not isinstance(n_win, np.ndarray) or n_win.dtype != np.int16 or n_win.shape != (n,):
            raise ValueError(f"N of window ts={ts} is not int16 [{n}] ({type(n_win).__name__} "
                             f"{getattr(n_win, 'dtype', '?')} {getattr(n_win, 'shape', '?')})")
        if n and (int(n_win.min()) < N_MIN or int(n_win.max()) > N_MAX):
            raise ValueError(f"N of window ts={ts} outside [{N_MIN}, {N_MAX}]: [{int(n_win.min())}, {int(n_win.max())}]")
        t_ms = _now_ms()
        fd = os.memfd_create(f"dlrmv4-m4otf-w{ts}", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        if not self.first_memfd:
            self.first_memfd = True
            _say(f"[m4otf] node={self.host} first memfd_create j={j} ts={ts} t={t_ms}")
        try:
            view = memoryview(np.ascontiguousarray(n_win).view(np.uint8))
            done = 0
            while done < view.nbytes:
                done += os.write(fd, view[done:])
            fcntl.fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL)
        except BaseException:
            os.close(fd)
            raise
        sec = time.perf_counter() - t0
        rate = sec / max(n, MIN_EST_ANCHORS)
        self.s_per_anchor = rate if self.s_per_anchor is None else 0.5 * self.s_per_anchor + 0.5 * rate
        ms = sec * 1e3
        self.seq += 1
        self.pub[self.seq] = [j, ts, n, ms, fd, 0]
        self.n_built += 1
        _say(f"[m4otf] node={self.host} j={j} ts={ts} anchors={n} build_ms={ms:.1f} progress={self.progress} "
             f"need={self._need(j)} pick={how} t={t_ms}")
        if self.duty < 1.0:
            self.rest_until = time.monotonic() + sec * (1.0 / self.duty - 1.0)

    def _accept(self) -> None:
        while True:
            try:
                s, _ = self.lsock.accept()
            except (BlockingIOError, InterruptedError):
                return
            except ConnectionAbortedError:
                continue
            why = check_peer(s, self.parent)
            if not why and self.max_clients and self.accepted >= self.max_clients:
                why = f"already serving {self.max_clients} local clients"
            if why:
                self.refused += 1
                if self.refused <= 3:
                    _say(f"[m4otf] node={self.host} refused a client ({why}) t={_now_ms()}")
                try:
                    s.close()
                except OSError:
                    pass
                continue
            self.accepted += 1
            s.setblocking(False)
            self.clients.append([s, 1])

    def _flush(self) -> bool:
        blocked = False
        for c in list(self.clients):
            sock = c[0]
            for seq in sorted(s for s in self.pub if s >= c[1]):
                e = self.pub[seq]
                try:
                    _send_fd(sock, _HDR.pack(e[1], e[0], e[2], e[3]), e[4])
                except (BlockingIOError, InterruptedError):
                    blocked = True
                    break
                except OSError:
                    self.clients.remove(c)
                    try:
                        sock.close()
                    except OSError:
                        pass
                    break
                e[5] += 1
                c[1] = seq + 1
        return blocked

    def _gc(self) -> None:
        for seq in [s for s, e in self.pub.items() if e[5] >= self.lws or e[0] <= self.progress]:
            os.close(self.pub.pop(seq)[4])

    def _wait(self, timeout: float) -> bool:
        ev = self._poller.poll(max(0.0, float(timeout)) * 1000.0)
        for fd, mask in ev:
            if mask & select.POLLNVAL:
                raise OSError(errno.EBADF, f"fd {fd} is not open (poll: POLLNVAL)")
        return bool(ev)

    def run(self) -> None:
        while True:
            now = time.monotonic()
            if self._pick()[0] is not None:
                timeout = 0.0
            elif self.blocked:
                timeout = 0.05
            elif self.rest_until > now:
                timeout = min(1.0, self.rest_until - now)
            else:
                timeout = 1.0
            try:
                self._wait(timeout)
            except InterruptedError:
                pass
            if not self._read_ctrl():
                return
            if self.watch_ppid and os.getppid() != self.watch_ppid:
                return
            self._accept()
            self.blocked = self._flush()
            self._gc()
            j, cursor, how = self._pick()
            if j is not None:
                self._build(j, cursor, how)
                self.blocked = self._flush()
                self._gc()

    def shutdown(self) -> None:
        for e in self.pub.values():
            try:
                os.close(e[4])
            except OSError:
                pass
        self.pub.clear()
        for c in self.clients:
            try:
                c[0].close()
            except OSError:
                pass
        for s in (self.lsock, self.ctrl):
            try:
                s.close()
            except OSError:
                pass
        _say(f"[m4otf] node={self.host} builder exit built={self.n_built} refused={self.refused} t={_now_ms()}")
