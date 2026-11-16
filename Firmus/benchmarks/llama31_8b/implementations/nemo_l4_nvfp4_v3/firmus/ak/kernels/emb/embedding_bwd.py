"""embedding_bwd: byte-exact word-embedding weight gradient.

Recipe arithmetic, per feature column (columns are independent):
    acc = +0.0 (bf16)
    for each chunk c the token occurs in, in increasing order:
        t = fp32 sum of its rows in c, in row order
        acc = bf16(float(acc) + float(bf16(t)))
    grad_weight[token] = acc ; tokens that never occur stay +0.0

A stable sort of the flattened [b, s] ids gives positions ordered by (token, row),
hence by (token, chunk).  A per-position head flag (token change or chunk change)
marks where one (token, chunk) group starts.

The 739 MB zero-fill only has to write rows the chain will NOT: a bitmap of the
tokens that appear is built on a side stream, and the selective fill runs there for
the whole main-stream sequence, one warp per CTA, so it gives back the warp slots the
latency-bound grouping/partial/chain kernels need, with evict-first stores so the
write stream does not flush the small metadata out of L2.  The group-partial buffer P
is kept in L2 by the same hints (evict-last stores, evict-first release loads).

This version removes the chain kernel's last serial dependency.  The partials kernel
already computes, for every group, whether it is the last group of its token (the
`last` flag behind the single-group fast path); it now also records the index one past
that group in a per-token array GEND.  The chain kernel therefore walks a token's
groups with a plain counted loop g0..GEND[tok]: no per-window token scan, no block
wide tl.min and its two barriers, no BCNT prefix scan.  The sixteen partial loads of a
window are independent and pipeline across windows instead of each window paying a
dependent load plus a barrier before its loads can issue.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prep_kernel(SIDX, PERM, NFLAG, LOC, BCNT, ROFF, MCOUNT, N, S, B, BLK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(MCOUNT + tl.arange(0, 1), tl.zeros([1], dtype=tl.int32))
    o = pid * BLK + tl.arange(0, BLK)
    m = o < N
    tok = tl.load(SIDX + o, mask=m, other=-1)
    prev = tl.load(SIDX + tl.maximum(o - 1, 0), mask=m, other=-1)
    r = tl.load(PERM + o, mask=m, other=0).to(tl.int32)
    rp = tl.load(PERM + tl.maximum(o - 1, 0), mask=m, other=0).to(tl.int32)
    flag = ((o == 0) | (tok != prev) | ((r >> 5) != (rp >> 5))) & m
    f = flag.to(tl.int32)
    loc = tl.cumsum(f, 0)
    tl.store(NFLAG + o, f, mask=m)
    tl.store(LOC + o, loc, mask=m)
    tl.store(ROFF + o, (r % S) * B + r // S, mask=m)
    tl.store(BCNT + pid + tl.arange(0, 1),
             tl.sum(f, 0) + tl.zeros([1], dtype=tl.int32))


@triton.jit
def _zero_int_kernel(PTR, N, ZB: tl.constexpr):
    o = tl.program_id(0) * ZB + tl.arange(0, ZB)
    z = tl.zeros([ZB], dtype=tl.int32)
    tl.store(PTR + o, z, mask=o < N)


@triton.jit
def _mark_kernel(INDICES, SEEN, N, BLK: tl.constexpr):
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = o < N
    tok = tl.load(INDICES + o, mask=m, other=0).to(tl.int32)
    tl.store(SEEN + tok, 1, mask=m)


@triton.jit
def _fill_kernel(SEEN, OUT, V, H, ROWS_PER_PROG, BF: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * ROWS_PER_PROG
    end = tl.minimum(start + ROWS_PER_PROG, V)
    c = tl.arange(0, BF)
    z = tl.zeros([BF], dtype=tl.bfloat16)
    cm = c < H
    r = start
    while r < end:
        s = tl.load(SEEN + r)
        if s == 0:
            tl.store(OUT + r * H + c, z, mask=cm, cache_modifier=".cs")
        r += 1


@triton.jit
def _partials_kernel(G, ROFF, NFLAG, LOC, BCNT, SIDX, P, MLIST, MCOUNT, GTOK, GEND, OUT, HID, N,
                     BLK: tl.constexpr, NB: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
                     CHUNK_POS: tl.constexpr, NORMZ: tl.constexpr = False):
    pid = tl.program_id(0)
    start = pid * CHUNK_POS
    end = tl.minimum(start + CHUNK_POS, N)
    if start < N:
        # Global group index of the first head at or after start.
        bi = start // BLK
        bids = tl.arange(0, NB)
        bc = tl.load(BCNT + bids, mask=bids < bi, other=0)
        base = tl.sum(bc, 0)
        if start % BLK != 0:
            base += tl.load(LOC + start - 1)
        p = start
        g = base
        while p < end:
            if tl.load(NFLAG + p) != 0:
                i32 = tl.arange(0, 32)
                nf = tl.load(NFLAG + p + 1 + i32, mask=(p + 1 + i32) < N, other=1)
                ln = tl.min(tl.where(nf != 0, i32 + 1, 32))
                tok = tl.load(SIDX + p).to(tl.int32)
                prev = tl.load(SIDX + tl.maximum(p - 1, 0)).to(tl.int32)
                first = (p == 0) | (tok != prev)
                nxt = tl.load(SIDX + p + ln, mask=(p + ln) < N, other=-1).to(tl.int32)
                last = ((p + ln) >= N) | (nxt != tok)
                single = first & last
                c0 = tl.arange(0, B0)
                c1 = B0 + tl.arange(0, B1)
                m1 = c1 < HID
                t0 = tl.zeros([B0], dtype=tl.float32)
                t1 = tl.zeros([B1], dtype=tl.float32)
                i = 0
                while i < ln:
                    r = tl.load(ROFF + p + i)
                    t0 += tl.load(G + r * HID + c0,
                                  eviction_policy="evict_first").to(tl.float32)
                    t1 += tl.load(G + r * HID + c1, mask=m1, other=0.0,
                                  eviction_policy="evict_first").to(tl.float32)
                    i += 1
                r0 = t0.to(tl.bfloat16)
                r1 = t1.to(tl.bfloat16)
                tl.store(GTOK + g + tl.arange(0, 1), tok + tl.zeros([1], dtype=tl.int32))
                if last:
                    # one past the token's last group: the chain's stop index
                    tl.store(GEND + tok + tl.arange(0, 1),
                             (g + 1) + tl.zeros([1], dtype=tl.int32))
                if single:
                    base_out = OUT + tok * HID
                    if NORMZ:
                        # written straight into main_grad: the shipped path ADDS this value to a zeroed main_grad,
                        # and +0.0 + -0.0 = +0.0, so the -0.0 bit pattern (0x8000) is stored as +0.0; every other
                        # value (NaN included) is unchanged.  The chain path below never yields -0.0 (its sum starts
                        # at +0.0), nor does the fill.
                        b0 = r0.to(tl.int16, bitcast=True)
                        r0 = tl.where(b0 == -32768, tl.zeros_like(b0), b0).to(tl.bfloat16, bitcast=True)
                        b1 = r1.to(tl.int16, bitcast=True)
                        r1 = tl.where(b1 == -32768, tl.zeros_like(b1), b1).to(tl.bfloat16, bitcast=True)
                    tl.store(base_out + c0, r0, eviction_policy="evict_first")
                    tl.store(base_out + c1, r1, mask=m1, eviction_policy="evict_first")
                else:
                    base_p = P + g * HID
                    tl.store(base_p + c0, r0, eviction_policy="evict_last")
                    tl.store(base_p + c1, r1, mask=m1, eviction_policy="evict_last")
                    if first:
                        idx = tl.atomic_add(MCOUNT + tl.arange(0, 1), tl.full([1], 1, tl.int32))
                        tl.store(MLIST + idx, tl.zeros([1], dtype=tl.int32) + g)
                g += 1
            p += 1


@triton.jit
def _chain_kernel(P, MLIST, MCOUNT, GTOK, GEND, OUT, HID,
                  BH: tl.constexpr, U: tl.constexpr, C: tl.constexpr):
    i = tl.program_id(0)
    fb = tl.program_id(1)
    nm = tl.load(MCOUNT)
    cols = fb * BH + tl.arange(0, BH)
    cm = cols < HID
    while i < nm:
        g0 = tl.load(MLIST + i)
        tok = tl.load(GTOK + g0)
        g1 = tl.load(GEND + tok)
        acc = tl.zeros([BH], dtype=tl.float32)
        g = g0
        while g < g1:
            for j in tl.static_range(U):
                pj = tl.load(P + (g + j) * HID + cols,
                             mask=cm & ((g + j) < g1), other=0.0,
                             eviction_policy="evict_first")
                acc = (acc + pj.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            g += U
        tl.store(OUT + tok * HID + cols, acc.to(tl.bfloat16), mask=cm,
                 eviction_policy="evict_first")
        i += C


class EmbeddingBwd:
    """__call__(grad_sbh, indices_bs, num_weights, out) -> out: writes every byte of
    the persistent [num_weights, h] bf16 buffer (untouched rows +0.0)."""

    def __init__(self, device="cuda"):
        self.device = device
        self._cache = {}
        try:
            self._side = torch.cuda.Stream(device=device)
        except Exception:
            self._side = None

    def _get(self, n, v, h, dev, nb):
        key = (n, v, h, nb)
        b = self._cache.get(key)
        if b is None:
            b = {
                "nflag": torch.empty(n, dtype=torch.int32, device=dev),
                "loc": torch.empty(n, dtype=torch.int32, device=dev),
                "bcnt": torch.zeros(nb, dtype=torch.int32, device=dev),
                "roff": torch.empty(n, dtype=torch.int32, device=dev),
                "P": torch.empty(n, h, dtype=torch.bfloat16, device=dev),
                "mlist": torch.empty(n, dtype=torch.int32, device=dev),
                "mcount": torch.empty(1, dtype=torch.int32, device=dev),
                "gtok": torch.empty(n, dtype=torch.int32, device=dev),
                "gend": torch.empty(v, dtype=torch.int32, device=dev),
                "seen": torch.empty(v, dtype=torch.int32, device=dev),
            }
            self._cache[key] = b
        return b

    def __call__(self, grad_sbh, indices_bs, num_weights, out, normz=False):
        """normz=True (2026-10-07, the main_grad target): a -0.0 single-group result is stored as +0.0, so `out`
        holds exactly what `zeroed main_grad += gradient` would."""
        s, b, h = grad_sbh.shape
        v = num_weights
        dev = grad_sbh.device
        n = s * b
        BLK = 1024
        nblk = triton.cdiv(n, BLK)
        NB = max(64, triton.next_power_of_2(nblk))
        buf = self._get(n, v, h, dev, NB)
        nflag = buf["nflag"]
        loc = buf["loc"]
        bcnt = buf["bcnt"]
        roff = buf["roff"]
        P = buf["P"]
        mlist = buf["mlist"]
        mcount = buf["mcount"]
        gtok = buf["gtok"]
        gend = buf["gend"]
        seen = buf["seen"]

        side = self._side
        cur = torch.cuda.current_stream()

        NPROG_FILL = 1024
        FBF = 4096
        rows_per_prog = triton.cdiv(v, NPROG_FILL)

        def _side_work():
            ZB_SEEN = 8192
            _zero_int_kernel[(triton.cdiv(v, ZB_SEEN),)](seen, v, ZB=ZB_SEEN, num_warps=4)
            MBLK = 1024
            _mark_kernel[(triton.cdiv(n, MBLK),)](indices_bs, seen, n, BLK=MBLK, num_warps=4)
            _fill_kernel[(NPROG_FILL,)](seen, out, v, h, rows_per_prog, BF=FBF, num_warps=1)

        if side is not None:
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                _side_work()
        else:
            _side_work()

        sidx, perm = torch.sort(indices_bs.reshape(-1).to(torch.int32), stable=True)

        _prep_kernel[(nblk,)](sidx, perm, nflag, loc, bcnt, roff, mcount,
                              n, s, b, BLK=BLK, num_warps=4)

        B0 = 2048
        # Llama 3.1 8B campaign (2026-10-02): the gpt-oss kernel covered h <= B0 + 1024 = 3072 columns (hidden 2880);
        # hidden 4096 left columns 3072-4095 unwritten.  For h > 3072 the second block grows to the next power of two
        # of the remainder (4096 -> 2048).  Each column is still summed in the same order, so the bytes are unchanged
        # wherever the old kernel covered the column; h <= 3072 keeps the old constants exactly.
        B1 = 1024 if h <= B0 + 1024 else triton.next_power_of_2(h - B0)
        CHUNK_POS = 16
        nprog = triton.cdiv(n, CHUNK_POS)
        _partials_kernel[(nprog,)](
            grad_sbh, roff, nflag, loc, bcnt, sidx, P, mlist, mcount, gtok, gend, out, h, n,
            BLK=BLK, NB=NB, B0=B0, B1=B1, CHUNK_POS=CHUNK_POS, NORMZ=bool(normz), num_warps=8)

        C = 1024
        _chain_kernel[(C, triton.cdiv(h, 512))](
            P, mlist, mcount, gtok, gend, out, h,
            BH=512, U=16, C=C, num_warps=4)

        if side is not None:
            cur.wait_stream(side)
        return out
