# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import operator

import numpy as np

__all__ = ["compute_window_n_incremental", "compute_window_n", "NState",
           "SCAN_WINDOW", "PER_POOL", "UIH_BUDGET", "N_EXTRA"]

SCAN_WINDOW = 20000
PER_POOL = 1362
UIH_BUDGET = 4087
N_EXTRA = 9
N_POOLS = 3


PIECE_ANCHORS = 1 << 18
CHUNK_EVENTS = 1 << 20
_STATE_VERSION = 1


def _as_positions(positions):
    p = np.asarray(positions)
    if p.ndim != 1:
        raise ValueError(f"positions must be 1-D, got shape {p.shape}")
    if p.dtype != np.int64:
        if p.dtype.kind not in "iu":
            raise TypeError(f"positions must be integers, got {p.dtype}")
        p = p.astype(np.int64)
    return p


def _as_user_start(user_start):
    u = np.asarray(user_start)
    if u.ndim != 1 or u.shape[0] < 2:
        raise ValueError(f"user_start must be 1-D with n_users + 1 >= 2 entries, got shape {u.shape}")
    if u.dtype != np.int64:
        if u.dtype.kind not in "iu":
            raise TypeError(f"user_start must be integers, got {u.dtype}")
        u = u.astype(np.int64)
    return u


def _as_flags(flags):
    if not isinstance(flags, (tuple, list)) or len(flags) != N_POOLS:
        raise ValueError("flags must be a tuple (flat_is_listen_plus, flat_is_like, flat_is_skip)")
    out = []
    for f in flags:
        a = np.asarray(f)
        if a.ndim != 1 or a.dtype.itemsize != 1 or a.dtype.kind not in "biu":
            raise ValueError(f"pool flags must be 1-D 1-byte arrays, got {a.dtype} ndim={a.ndim}")
        out.append(a.view(np.uint8))
    if any(a.shape[0] != out[0].shape[0] for a in out):
        raise ValueError("pool flag columns differ in length")
    return tuple(out)


def _params(scan_window, per_pool, uih_budget, n_extra):
    w, pp, ub, ne = (operator.index(x) for x in (scan_window, per_pool, uih_budget, n_extra))
    if min(w, pp, ub, ne) < 0:
        raise ValueError("scan_window, per_pool, uih_budget and n_extra must be >= 0")
    if w >= 1 << 29:
        raise ValueError("scan_window must be < 2**29 (int32 counts)")
    if ub + ne > np.iinfo(np.int16).max:
        raise ValueError("uih_budget + n_extra must fit int16")
    return (w, pp, ub, ne)


def _is_ascending(p):
    return p.shape[0] < 2 or bool(np.all(p[1:] >= p[:-1]))


def _stream_prefix(flags, lo, ln, queries, chunk_events, stats=None):
    keep = np.flatnonzero(ln > 0)
    if keep.shape[0] == 0:
        return [[np.zeros(q.shape[0], np.int64) for q in queries] for _ in flags], 0

    res = [[np.empty(q.shape[0], np.int64) for q in queries] for _ in flags]
    plo = lo[keep]
    pln = ln[keep]
    pend = np.cumsum(pln)
    total = int(pend[-1])
    poff = pend - pln
    delta = plo - poff
    cap = min(int(chunk_events), total)
    idx = np.empty(cap, np.int64)
    buf = np.empty(cap, np.uint8)
    cs = np.empty(cap + 1, np.int32)
    carry = [0] * len(flags)
    nchunks = 0
    for c0 in range(0, total, cap):
        c1 = min(total, c0 + cap)
        k = c1 - c0
        last = c1 == total
        k0 = int(np.searchsorted(pend, c0, side="right"))
        k1 = int(np.searchsorted(poff, c1, side="left"))

        ix = idx[:k]
        ix.fill(1)
        starts = poff[k0 + 1:k1] - c0
        ix[starts] = delta[k0 + 1:k1] - delta[k0:k1 - 1] + 1
        ix[0] = c0 + delta[k0]
        np.cumsum(ix, out=ix)
        sel = []
        for q in queries:
            i0 = int(np.searchsorted(q, c0, side="left"))
            i1 = q.shape[0] if last else int(np.searchsorted(q, c1, side="left"))
            sel.append((i0, i1, q[i0:i1] - c0 if i1 > i0 else None))
        b = buf[:k]
        c = cs[:k + 1]
        c[0] = 0
        for f, fl in enumerate(flags):
            np.take(fl, ix, out=b, mode="clip")
            np.cumsum(b, dtype=np.int32, out=c[1:])
            for (i0, i1, rel), out in zip(sel, res[f]):
                if rel is not None:
                    v = out[i0:i1]
                    v[...] = c[rel]
                    v += carry[f]
            carry[f] += int(c[k])
        nchunks += 1
    if stats is not None:
        stats["chunks"] = stats.get("chunks", 0) + nchunks
    return res, total


def _n_from_counts(counts_iter, m, params):
    _, per_pool, budget, extra = params
    uih = np.zeros(m, np.int32)
    for c in counts_iter:
        np.minimum(c, per_pool, out=c)
        uih += c
    np.minimum(uih, budget, out=uih)
    uih += extra
    return uih.astype(np.int16)


class NState:
    __slots__ = ("version", "user_lo", "user_hi", "n_users", "n_events", "guard", "params",
                 "last_p", "last_s", "cnt", "calls", "busy")

    def __init__(self, user_lo, user_hi, user_start, n_events, params):
        n = user_hi - user_lo
        self.version = _STATE_VERSION
        self.user_lo = user_lo
        self.user_hi = user_hi
        self.n_users = int(user_start.shape[0]) - 1
        self.n_events = int(n_events)
        self.guard = (int(user_start[0]), int(user_start[user_lo]), int(user_start[user_hi]),
                      int(user_start[-1]))
        self.params = params
        self.last_p = np.full(n, -1, np.int64)
        self.last_s = np.zeros(n, np.int64)
        self.cnt = np.zeros((N_POOLS, n), np.int32)
        self.calls = 0
        self.busy = False

    @property
    def nbytes(self):
        return int(self.last_p.nbytes + self.last_s.nbytes + self.cnt.nbytes)

    def _check(self, user_lo, user_hi, user_start, n_events, params):
        if self.version != _STATE_VERSION:
            raise ValueError("state from another m4_nfunc version")
        if self.busy:
            raise ValueError("state was left inside a failed call; restart with state=None")
        got = (user_lo, user_hi, int(user_start.shape[0]) - 1, int(n_events), params,
               (int(user_start[0]), int(user_start[user_lo]), int(user_start[user_hi]), int(user_start[-1])))
        want = (self.user_lo, self.user_hi, self.n_users, self.n_events, self.params, self.guard)
        if got != want:
            raise ValueError(f"state does not match this call (user range / user_start / flags / parameters): "
                             f"state {want} vs call {got}")


def _uid_sorted(ps, ust):
    m = ps.shape[0]
    u0 = int(np.searchsorted(ust, ps[0], side="right")) - 1
    u1 = int(np.searchsorted(ust, ps[m - 1], side="right")) - 1
    if u1 == u0:
        return np.full(m, u0, np.int64)
    if u1 - u0 >= m:
        uid = np.searchsorted(ust[u0:u1 + 1], ps, side="right")
        uid += u0 - 1
        return uid.astype(np.int64, copy=False)
    bnd = np.searchsorted(ps, ust[u0 + 1:u1 + 1], side="left")
    uid = np.cumsum(np.bincount(bnd, minlength=m)[:m])
    uid += u0
    return uid.astype(np.int64, copy=False)


def _piece_incremental(ps, ust, fl, st, params, chunk_events, stats):
    w = params[0]
    m = ps.shape[0]
    uid = _uid_sorted(ps, ust)
    s = ust[uid]
    np.maximum(s, ps - w, out=s)
    chg = np.flatnonzero(uid[1:] != uid[:-1])
    chg += 1
    gf = np.concatenate((np.zeros(1, np.int64), chg))
    gl = np.concatenate((chg - 1, np.full(1, m - 1, np.int64)))
    gsz = gl - gf + 1
    gu = uid[gf] - st.user_lo
    del uid, chg
    ng = gf.shape[0]
    pf = ps[gf]
    sf = s[gf]
    pl = ps[gl]
    sl = s[gl]
    sp = st.last_p[gu]
    ss = st.last_s[gu]
    inc = sp >= 0
    inc &= pf >= sp
    inc &= sf >= ss
    inc &= (pl - sp) + (sl - ss) < (pl - sf)


    lo1 = np.where(inc, ss, sf)
    hi1 = np.where(inc, sl, pl)
    lo2 = np.where(inc, sp, pl)
    lo = np.empty(2 * ng, np.int64)
    ln = np.empty(2 * ng, np.int64)
    lo[0::2] = lo1
    lo[1::2] = lo2
    ln[0::2] = hi1 - lo1
    ln[1::2] = pl - lo2
    off = np.empty(2 * ng + 1, np.int64)
    off[0] = 0
    np.cumsum(ln, out=off[1:])
    off1 = off[0:2 * ng:2]
    off2 = off[1:2 * ng:2]

    plus = np.repeat(np.where(inc, off2 - lo2, off1 - lo1), gsz)
    plus += ps
    minus = np.repeat(off1 - lo1, gsz)
    minus += s
    inc_idx = np.flatnonzero(inc)
    queries = (plus, minus, off1[inc_idx], off2[inc_idx])
    pref, total = _stream_prefix(fl, lo, ln, queries, chunk_events, stats)
    del plus, minus
    span = ps - s
    new_cnt = np.empty((N_POOLS, ng), np.int32)
    counts = []
    gu_inc = gu[inc_idx]
    for k in range(N_POOLS):
        gp, gm, g1, g2 = pref[k]
        base = np.zeros(ng, np.int64)
        base[inc_idx] = st.cnt[k, gu_inc].astype(np.int64) + g1 - g2
        c = np.repeat(base, gsz)
        c += gp
        c -= gm
        if m and (int(c.min()) < 0 or bool(np.any(c > span))):
            raise ValueError("inconsistent counts: the state does not match user_start/flags")
        new_cnt[k] = c[gl]
        counts.append(c)
        pref[k] = None
    out = _n_from_counts(counts, m, params)
    st.last_p[gu] = pl
    st.last_s[gu] = sl
    st.cnt[:, gu] = new_cnt
    if stats is not None:
        n_inc = int(inc_idx.shape[0])
        ev_inc = int(ln[0::2][inc_idx].sum() + ln[1::2][inc_idx].sum()) if n_inc else 0
        stats["pieces"] = stats.get("pieces", 0) + 1
        stats["groups"] = stats.get("groups", 0) + ng
        stats["groups_incremental"] = stats.get("groups_incremental", 0) + n_inc
        stats["groups_rebuild"] = stats.get("groups_rebuild", 0) + ng - n_inc
        stats["stream_events"] = stats.get("stream_events", 0) + total
        stats["stream_events_incremental"] = stats.get("stream_events_incremental", 0) + ev_inc
        stats["stream_events_rebuild"] = stats.get("stream_events_rebuild", 0) + total - ev_inc
    return out


def compute_window_n_incremental(positions, user_start, flags, state, *, user_lo, user_hi,
                                 scan_window=SCAN_WINDOW, per_pool=PER_POOL, uih_budget=UIH_BUDGET,
                                 n_extra=N_EXTRA, stats=None, piece_anchors=PIECE_ANCHORS,
                                 chunk_events=CHUNK_EVENTS):


    p = _as_positions(positions)
    ust = _as_user_start(user_start)
    fl = _as_flags(flags)
    params = _params(scan_window, per_pool, uih_budget, n_extra)
    piece_anchors = operator.index(piece_anchors)
    chunk_events = operator.index(chunk_events)
    if piece_anchors < 1 or chunk_events < 1:
        raise ValueError("piece_anchors and chunk_events must be >= 1")
    n_users = ust.shape[0] - 1
    user_lo = operator.index(user_lo)
    user_hi = operator.index(user_hi)
    if not 0 <= user_lo < user_hi <= n_users:
        raise ValueError(f"need 0 <= user_lo < user_hi <= n_users={n_users}, got {user_lo}, {user_hi}")
    n_events = fl[0].shape[0]
    if int(ust[0]) < 0 or int(ust[-1]) > n_events:
        raise ValueError("user_start does not lie inside the flag columns")
    if state is None:
        state = NState(user_lo, user_hi, ust, n_events, params)
    elif not isinstance(state, NState):
        raise TypeError(f"state must be None or the NState this function returned, got {type(state).__name__}")
    else:
        state._check(user_lo, user_hi, ust, n_events, params)
    n = p.shape[0]
    out = np.empty(n, np.int16)
    if stats is not None:
        stats["anchors"] = stats.get("anchors", 0) + n
    if n == 0:
        state.calls += 1
        return out, state
    if _is_ascending(p):
        order = None
        ps = p
    else:
        order = np.argsort(p, kind="stable")
        ps = p[order]
        if stats is not None:
            stats["unsorted_input"] = stats.get("unsorted_input", 0) + 1
    if int(ps[0]) < int(ust[user_lo]) or int(ps[n - 1]) >= int(ust[user_hi]):
        raise ValueError(f"anchor position outside the events of users [{user_lo}, {user_hi}): "
                         f"[{int(ps[0])}, {int(ps[n - 1])}] vs [{int(ust[user_lo])}, {int(ust[user_hi])})")
    state.busy = True
    for a in range(0, n, piece_anchors):
        b = min(n, a + piece_anchors)
        r = _piece_incremental(ps[a:b], ust, fl, state, params, chunk_events, stats)
        if order is None:
            out[a:b] = r
        else:
            out[order[a:b]] = r
    state.busy = False
    state.calls += 1
    return out, state


def compute_window_n(positions, user_start, flags, *, scan_window=SCAN_WINDOW, per_pool=PER_POOL,
                     uih_budget=UIH_BUDGET, n_extra=N_EXTRA, piece_anchors=PIECE_ANCHORS,
                     chunk_events=CHUNK_EVENTS):


    p = _as_positions(positions)
    ust = _as_user_start(user_start)
    fl = _as_flags(flags)
    params = _params(scan_window, per_pool, uih_budget, n_extra)
    w = params[0]
    n = p.shape[0]
    out = np.empty(n, np.int16)
    if n == 0:
        return out
    if int(ust[0]) < 0 or int(ust[-1]) > fl[0].shape[0]:
        raise ValueError("user_start does not lie inside the flag columns")
    order = np.argsort(p, kind="stable")
    ps = p[order]
    uid = np.searchsorted(ust, ps, side="right") - 1
    if int(uid[0]) < 0 or int(uid[-1]) >= ust.shape[0] - 1:
        raise ValueError("anchor position outside [user_start[0], user_start[-1])")
    s = np.maximum(ust[uid], ps - w)
    del uid
    for a in range(0, n, piece_anchors):
        b = min(n, a + piece_anchors)
        pp = ps[a:b]
        sp = s[a:b]
        m = b - a


        newseg = np.empty(m, bool)
        newseg[0] = True
        np.greater(sp[1:], pp[:-1], out=newseg[1:])
        first = np.flatnonzero(newseg)
        last = np.empty_like(first)
        last[:-1] = first[1:] - 1
        last[-1] = m - 1
        seg_lo = sp[first]
        seg_ln = pp[last] - seg_lo
        seg_off = np.cumsum(seg_ln) - seg_ln
        shift = np.repeat(seg_off - seg_lo, last - first + 1)
        plus = pp + shift
        minus = sp + shift
        del shift
        pref, _ = _stream_prefix(fl, seg_lo, seg_ln, (plus, minus), chunk_events)
        counts = []
        for k in range(N_POOLS):
            gp, gm = pref[k]
            gp -= gm
            counts.append(gp)
            pref[k] = None
        out[order[a:b]] = _n_from_counts(counts, m, params)
    return out
