"""(x9step3, 2026-10-07: the step kernels can also fold each shard's max |bf16 param| into a weight-amax buffer.)
FTZ + true -lr negation (ftz_e) applied to the x9FOLD kernel (2ba8b61a); the norm kernels unchanged: the fp32 chain as inline PTX (div.full, sqrt.approx), everything else K.

Byte-exact fused AdamW master step WITH TE's gradient-norm kernel (Triton) -- whole-file replacement.

The optimizer_adamw winner's step (streaming AdamW-master: 28 B/element once, 4 elements per
thread, 128-bit fp32 / 64-bit bf16 accesses) plus TE's ``multi_tensor_l2norm`` order for the
local norm.  ``norm(grads)`` returns TE's bytes; Megatron squares, all-reduces, roots and clips
outside this module.  ``step(..., clip_coeff, ...)`` is the optimizer contract.

THE NORM IS A FIXED REDUCTION ORDER, NOT A VALUE.  TE's kernel (BLOCK_SIZE 512, ILP 4, chunk
65536, at most 320 blocks and 110 tensors per launch) is reproduced step by step:

  1. one block per 65536-element CHUNK of a shard; 512 lanes; lane t keeps four fp32
     accumulators vals[0..3] (start 0):
       aligned path  (numel % 4 == 0 -- the shipped chain norms a fresh, aligned fp32 copy):
            for j = 0, 1, ...:  e = 4 * (t + 512 j);  vals[ii] = fma(x[e+ii], x[e+ii], vals[ii])
       otherwise:
            for j = 0, 1, ...:  e = 2048 j + t + 512 ii;  vals[ii] = fma(x[e], x[e], vals[ii])
       (x is float32(bf16); elements past the chunk or the shard contribute nothing);
     val = ((vals[0] + vals[1]) + vals[2]) + vals[3]
  2. the block tree over the 512 lane values, pairs (t, t + 256), (t, t + 128), ..., (0, 1);
  3. the block result is written to slot ``blockIdx.x`` of a 320-entry fp32 buffer; a launch is
     cut after 320 blocks, after the last chunk of the 110th tensor, or at the very end;
  4. cleanup: lane t < 320 takes slot t, lanes 320..511 take 0, the same tree,
     grad_norm = sqrt(total)   (``sqrt.approx.f32``: TE's common library is a fast-math build).

LAUNCH GEOMETRY.  One grid carries every chunk of the round; each program's class is a field of
its table row, so a single launch covers all three shapes and the norm costs exactly two
launches (chunk + finish).  Sequential grids on one stream cannot overlap -- each one ramps up
and drains before the next starts -- so folding the classes into one grid removes two ramps and
two drains from the front of the round while keeping every slot write disjoint.

  * class 0 -- complete chunks of a 32-byte aligned shard, up to ``_L2_NCH`` per program, with
    unmasked wide loads;
  * class 1 -- the ragged TAIL chunk of a 32-byte aligned shard, one program per tail, with the
    same wide loads.  ``rem < 65536`` and only the last iteration can straddle the shard end,
    so the loop is an unmasked prefix (``j < rem // 2048``) plus one masked iteration -- the fma
    chain over j, the ((a0+a1)+a2)+a3 combination and the (launch, slot) cell are TE's, untouched;
  * class 2 -- everything else (shards whose base is not 32-byte aligned), the masked kernel
    verbatim.

The lane order fixes WHICH elements a lane squares and in what order, not how the block's loads
are issued.  ``tl.multiple_of`` on a pointer is a BYTE count, so the classes' hints are 16: the
class-0/1 base is 32-byte aligned by construction and a chunk stride is 131072 bytes, hence a
lane's 4-group is one 64-bit word (strided path) and two 128-bit words (aligned path).

TABLES.  Both passes cache their tables on the *list objects* the training loop hands in, so the
per-call Python cost is a handful of identity tests -- the round's first launch is not delayed by
a 418-entry ``data_ptr()`` walk while the GPU sits idle.

The step kernels read the clip coefficient from the device tensor Megatron hands in -- nothing
touches the host.
"""

import math

import numpy as np
import torch
import triton
import triton.language as tl


# ---- T3 (2026-09-28): the per-element fp32 chain as inline PTX, FTZ = ".ftz" ----------------------
@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise("mul.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _fma(a, b, c):
    return tl.inline_asm_elementwise("fma.rn.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _add(a, b):
    return tl.inline_asm_elementwise("add.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _div(a, b):
    return tl.inline_asm_elementwise("div.full.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _sqrt(a):
    return tl.inline_asm_elementwise("sqrt.approx.ftz.f32 $0, $1;", "=f,f", [a], dtype=tl.float32,
                                     is_pure=True, pack=1)
# ------------------------------------------------------------------------------------------------------

_BLOCK = 512       # elements per program iteration -- exactly 4 per thread
_NBLK = 8          # consecutive full blocks walked by one bulk program
_NUM_WARPS = 4     # 128 threads -> 4 elements per thread on every stream

_L2_CHUNK = 65536  # TE's chunk: 2048 * 32 elements
_L2_LANES = 512    # TE's BLOCK_SIZE
_L2_SLOTS = 320    # TE's blocks per launch = slots of the partial buffer
_L2_MAX_T = 110    # TE's tensors per launch (depth 1)
_L2_NCH = 4        # TE chunks handled by one program of the norm's chunk grid


def host_scalars(beta1, beta2, step, bias_correction=True):
    """TE's host-side constants, as the CUDA kernel receives them (float32).

    ``beta`` is rounded to float32 first (that is the value TE's kernel sees);
    the bias corrections are ``1 - beta**step`` evaluated in double on that
    float32 beta and rounded once to float32, exactly like the reference.
    """
    b1 = np.float32(beta1)
    b2 = np.float32(beta2)
    omb1 = np.float32(1.0) - b1
    omb2 = np.float32(1.0) - b2
    if bias_correction:
        bc1 = np.float32(1.0 - math.pow(float(b1), step))
        bc2 = np.float32(1.0 - math.pow(float(b2), step))
    else:
        bc1 = np.float32(1.0)
        bc2 = np.float32(1.0)
    return float(b1), float(omb1), float(b2), float(omb2), float(bc1), float(bc2)


@triton.jit
def _tree512(v):
    """TE's reduce_block_into_lanes over 512 lane values: (t, t+256), (t, t+128), ... (0, 1).
    Every level is a sum of exactly two numbers, so the result is the CUDA kernel's bytes."""
    v = tl.sum(tl.reshape(v, [2, 256]), axis=0)
    v = tl.sum(tl.reshape(v, [2, 128]), axis=0)
    v = tl.sum(tl.reshape(v, [2, 64]), axis=0)
    v = tl.sum(tl.reshape(v, [2, 32]), axis=0)
    v = tl.sum(tl.reshape(v, [2, 16]), axis=0)
    v = tl.sum(tl.reshape(v, [2, 8]), axis=0)
    v = tl.sum(tl.reshape(v, [2, 4]), axis=0)
    v = tl.sum(tl.reshape(v, [2, 2]), axis=0)
    v = tl.sum(tl.reshape(v, [2, 1]), axis=0)
    return v


@triton.jit
def _l2norm_chunk_kernel(addr_ptr, numel_ptr, aligned_ptr, kind_ptr, tob_ptr, c0_ptr, nch_ptr,
                         gidx_ptr, dest_ptr, partial_ptr,
                         CHUNK: tl.constexpr, LANES: tl.constexpr):
    """Every TE chunk of the round in ONE grid.  ``kind`` selects the program class; the slot
    writes of the three classes are disjoint, so the (launch, slot) cells -- and therefore the
    bytes -- are exactly the three-grid version's.

    kind 0: up to ``nch`` consecutive COMPLETE chunks of one 32-byte aligned shard.  Nothing is
            masked, and the base is 32-byte aligned (chunk strides are multiples of 131072
            bytes), so the pointer hint is 16 BYTES: the aligned path fetches lane t's 4-group
            as one 64-bit word, the strided path fetches four adjacent lanes' 2-byte elements
            as one word.
    kind 1: the single ragged TAIL chunk of a 32-byte aligned shard, same wide loads.
    kind 2: every chunk of a shard whose base is not 32-byte aligned, masked scalar loads.

    The lane order, the fma chain over j, the ((a0+a1)+a2)+a3 combination and the block tree are
    TE's, unchanged in all three."""
    pid = tl.program_id(0)
    kind = tl.load(kind_ptr + pid)
    t = tl.load(tob_ptr + pid)
    c0 = tl.load(c0_ptr + pid)
    nch = tl.load(nch_ptr + pid)
    gidx = tl.load(gidx_ptr + pid)
    g_ptr = tl.cast(tl.load(addr_ptr + t), tl.pointer_type(tl.bfloat16))
    lanes = tl.arange(0, LANES)

    if kind == 0:
        aligned = tl.load(aligned_ptr + t)
        for k in range(nch):
            c = c0 + k
            dest = tl.load(dest_ptr + gidx + k)   # launch * 320 + slot
            base = g_ptr + c * CHUNK
            acc0 = tl.zeros([LANES], dtype=tl.float32)
            acc1 = tl.zeros([LANES], dtype=tl.float32)
            acc2 = tl.zeros([LANES], dtype=tl.float32)
            acc3 = tl.zeros([LANES], dtype=tl.float32)
            if aligned != 0:
                # lane t owns the 4-groups at 4 * (t + 512 j); ILP slot ii = element within the group
                g64 = tl.multiple_of(tl.cast(base, tl.pointer_type(tl.uint64)), 16)
                for j in range(CHUNK // (4 * LANES)):
                    w = tl.load(g64 + j * LANES + lanes)
                    lo = w.to(tl.int32)               # elements 4i+0 (low half) and 4i+1 (high half)
                    hi = (w >> 32).to(tl.int32)       # elements 4i+2 (low half) and 4i+3 (high half)
                    x0 = (lo << 16).to(tl.float32, bitcast=True)
                    x1 = ((lo >> 16) << 16).to(tl.float32, bitcast=True)
                    x2 = (hi << 16).to(tl.float32, bitcast=True)
                    x3 = ((hi >> 16) << 16).to(tl.float32, bitcast=True)
                    acc0 = tl.fma(x0, x0, acc0)
                    acc1 = tl.fma(x1, x1, acc1)
                    acc2 = tl.fma(x2, x2, acc2)
                    acc3 = tl.fma(x3, x3, acc3)
            else:
                # lane t, ILP slot ii owns element 2048 j + t + 512 ii
                gf = tl.multiple_of(base, 16)
                for j in range(CHUNK // (4 * LANES)):
                    e = j * (4 * LANES) + lanes
                    x0 = tl.load(gf + e).to(tl.float32)
                    x1 = tl.load(gf + e + LANES).to(tl.float32)
                    x2 = tl.load(gf + e + 2 * LANES).to(tl.float32)
                    x3 = tl.load(gf + e + 3 * LANES).to(tl.float32)
                    acc0 = tl.fma(x0, x0, acc0)
                    acc1 = tl.fma(x1, x1, acc1)
                    acc2 = tl.fma(x2, x2, acc2)
                    acc3 = tl.fma(x3, x3, acc3)
            val = ((acc0 + acc1) + acc2) + acc3
            total = _tree512(val)
            tl.store(partial_ptr + dest + tl.arange(0, 1), total)

    elif kind == 1:
        # ``rem < CHUNK``, so the iterations in which every lane is in bounds are
        # ``nfull = rem // (4 * LANES)``; only iteration ``j = nfull`` can straddle the shard
        # end.  The loop is an unmasked prefix plus one masked iteration -- the accumulator
        # order over j, the ((a0+a1)+a2)+a3 combination, the block tree and the (launch, slot)
        # cell are the scalar kernel's, unchanged.
        n = tl.load(numel_ptr + t)
        aligned = tl.load(aligned_ptr + t)
        rem = n - c0 * CHUNK
        dest = tl.load(dest_ptr + gidx)           # launch * 320 + slot
        base = g_ptr + c0 * CHUNK
        nfull = rem // (4 * LANES)
        acc0 = tl.zeros([LANES], dtype=tl.float32)
        acc1 = tl.zeros([LANES], dtype=tl.float32)
        acc2 = tl.zeros([LANES], dtype=tl.float32)
        acc3 = tl.zeros([LANES], dtype=tl.float32)
        if aligned != 0:
            # lane t owns the 4-groups at 4 * (t + 512 j); ILP slot ii = element within the group
            g64 = tl.multiple_of(tl.cast(base, tl.pointer_type(tl.uint64)), 16)
            for j in range(nfull):
                w = tl.load(g64 + j * LANES + lanes)
                lo = w.to(tl.int32)
                hi = (w >> 32).to(tl.int32)
                x0 = (lo << 16).to(tl.float32, bitcast=True)
                x1 = ((lo >> 16) << 16).to(tl.float32, bitcast=True)
                x2 = (hi << 16).to(tl.float32, bitcast=True)
                x3 = ((hi >> 16) << 16).to(tl.float32, bitcast=True)
                acc0 = tl.fma(x0, x0, acc0)
                acc1 = tl.fma(x1, x1, acc1)
                acc2 = tl.fma(x2, x2, acc2)
                acc3 = tl.fma(x3, x3, acc3)
            if nfull * (4 * LANES) < rem:
                m = (nfull * LANES + lanes) * 4 < rem
                w = tl.load(g64 + nfull * LANES + lanes, mask=m, other=0)
                lo = w.to(tl.int32)
                hi = (w >> 32).to(tl.int32)
                x0 = (lo << 16).to(tl.float32, bitcast=True)
                x1 = ((lo >> 16) << 16).to(tl.float32, bitcast=True)
                x2 = (hi << 16).to(tl.float32, bitcast=True)
                x3 = ((hi >> 16) << 16).to(tl.float32, bitcast=True)
                acc0 = tl.fma(x0, x0, acc0)
                acc1 = tl.fma(x1, x1, acc1)
                acc2 = tl.fma(x2, x2, acc2)
                acc3 = tl.fma(x3, x3, acc3)
        else:
            # lane t, ILP slot ii owns element 2048 j + t + 512 ii
            gf = tl.multiple_of(base, 16)
            for j in range(nfull):
                e = j * (4 * LANES) + lanes
                x0 = tl.load(gf + e).to(tl.float32)
                x1 = tl.load(gf + e + LANES).to(tl.float32)
                x2 = tl.load(gf + e + 2 * LANES).to(tl.float32)
                x3 = tl.load(gf + e + 3 * LANES).to(tl.float32)
                acc0 = tl.fma(x0, x0, acc0)
                acc1 = tl.fma(x1, x1, acc1)
                acc2 = tl.fma(x2, x2, acc2)
                acc3 = tl.fma(x3, x3, acc3)
            if nfull * (4 * LANES) < rem:
                e = nfull * (4 * LANES) + lanes
                x0 = tl.load(gf + e, mask=e < rem, other=0.0).to(tl.float32)
                x1 = tl.load(gf + e + LANES, mask=(e + LANES) < rem, other=0.0).to(tl.float32)
                x2 = tl.load(gf + e + 2 * LANES, mask=(e + 2 * LANES) < rem, other=0.0).to(tl.float32)
                x3 = tl.load(gf + e + 3 * LANES, mask=(e + 3 * LANES) < rem, other=0.0).to(tl.float32)
                acc0 = tl.fma(x0, x0, acc0)
                acc1 = tl.fma(x1, x1, acc1)
                acc2 = tl.fma(x2, x2, acc2)
                acc3 = tl.fma(x3, x3, acc3)
        val = ((acc0 + acc1) + acc2) + acc3
        total = _tree512(val)
        tl.store(partial_ptr + dest + tl.arange(0, 1), total)

    else:
        # shard whose base is not 32-byte aligned: the masked scalar kernel verbatim
        n = tl.load(numel_ptr + t)
        aligned = tl.load(aligned_ptr + t)
        for k in range(nch):
            c = c0 + k
            rem = n - c * CHUNK                 # elements left in the shard from this chunk on
            dest = tl.load(dest_ptr + gidx + k)  # launch * 320 + slot
            base = g_ptr + c * CHUNK
            acc0 = tl.zeros([LANES], dtype=tl.float32)
            acc1 = tl.zeros([LANES], dtype=tl.float32)
            acc2 = tl.zeros([LANES], dtype=tl.float32)
            acc3 = tl.zeros([LANES], dtype=tl.float32)
            if aligned != 0:
                # lane t owns the 4-groups at 4 * (t + 512 j); ILP slot ii = element within the group
                for j in range(CHUNK // (4 * LANES)):
                    e = (j * LANES + lanes) * 4
                    m = e < rem
                    x0 = tl.load(base + e + 0, mask=m, other=0.0).to(tl.float32)
                    x1 = tl.load(base + e + 1, mask=m, other=0.0).to(tl.float32)
                    x2 = tl.load(base + e + 2, mask=m, other=0.0).to(tl.float32)
                    x3 = tl.load(base + e + 3, mask=m, other=0.0).to(tl.float32)
                    acc0 = tl.fma(x0, x0, acc0)
                    acc1 = tl.fma(x1, x1, acc1)
                    acc2 = tl.fma(x2, x2, acc2)
                    acc3 = tl.fma(x3, x3, acc3)
            else:
                # lane t, ILP slot ii owns element 2048 j + t + 512 ii
                for j in range(CHUNK // (4 * LANES)):
                    e = j * (4 * LANES) + lanes
                    x0 = tl.load(base + e, mask=e < rem, other=0.0).to(tl.float32)
                    x1 = tl.load(base + e + LANES, mask=(e + LANES) < rem, other=0.0).to(tl.float32)
                    x2 = tl.load(base + e + 2 * LANES, mask=(e + 2 * LANES) < rem, other=0.0).to(tl.float32)
                    x3 = tl.load(base + e + 3 * LANES, mask=(e + 3 * LANES) < rem, other=0.0).to(tl.float32)
                    acc0 = tl.fma(x0, x0, acc0)
                    acc1 = tl.fma(x1, x1, acc1)
                    acc2 = tl.fma(x2, x2, acc2)
                    acc3 = tl.fma(x3, x3, acc3)
            val = ((acc0 + acc1) + acc2) + acc3
            total = _tree512(val)
            tl.store(partial_ptr + dest + tl.arange(0, 1), total)


@triton.jit
def _l2norm_finish_kernel(partial_ptr, nblk_ptr, nlaunch, norm_ptr,
                          SLOTS: tl.constexpr, LANES: tl.constexpr):
    """TE's slot accumulation over the launches (in launch order), its cleanup tree and sqrt.
    One program, 512 lanes.  What follows the norm (square, all-reduce, root, clip) is Megatron's."""
    lanes = tl.arange(0, LANES)
    acc = tl.zeros([LANES], dtype=tl.float32)
    for L in range(nlaunch):
        nb = tl.load(nblk_ptr + L)
        p = tl.load(partial_ptr + L * SLOTS + lanes, mask=(lanes < SLOTS) & (lanes < nb), other=0.0)
        acc = acc + p                          # output[slot] += final, launch by launch
    total = _tree512(acc)
    norm = tl.sqrt(total)                      # cleanup: *ret = sqrt(final) -- sqrt.approx.f32 (TE's common library is a fast-math build)
    tl.store(norm_ptr + tl.arange(0, 1), norm)


@triton.jit
def _adamw_bulk_kernel(addr_ptr, numel_ptr, tob_ptr, sblk_ptr, clip_ptr,
                       lr, beta1, omb1, beta2, omb2, bc1, bc2, eps, decay, slot_ptr, amax_ptr,
                       BLOCK: tl.constexpr, NBLK: tl.constexpr, ALIGN: tl.constexpr, AMAX: tl.constexpr):
    """_NBLK consecutive full BLOCK-sized blocks of one shard per program; no mask.  AMAX (x9step3): also the
    max |bf16 param| of the program's elements, folded into amax_ptr[slot of the shard] (int32 view of an fp32
    buffer) by atomicMax -- multi_amax_bf16's arithmetic on the integer bit patterns: |x| = bits & 0x7FFF, a NaN
    (> 0x7F80) counts as 0 (fmax ignores NaN), subnormals and infinities kept (no flush to zero)."""
    pid = tl.program_id(0)
    t = tl.load(tob_ptr + pid)
    n = tl.load(numel_ptr + t)
    nfull = n // BLOCK
    sblk = tl.load(sblk_ptr + pid)          # first full block this program owns
    g_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 0), tl.pointer_type(tl.bfloat16))
    pm_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 1), tl.pointer_type(tl.float32))
    m_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 2), tl.pointer_type(tl.float32))
    v_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 3), tl.pointer_type(tl.float32))
    p_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 4), tl.pointer_type(tl.bfloat16))
    clip = tl.load(clip_ptr)

    lanes = tl.arange(0, BLOCK)
    amx = tl.zeros([BLOCK], dtype=tl.int32)
    for j in range(NBLK):
        blk = sblk + j
        if blk < nfull:                     # scalar, uniform branch: no lane predication
            offs = blk * BLOCK + lanes
            offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
            r_g = _mul(tl.load(tl.multiple_of(g_ptr + offs, ALIGN)).to(tl.float32), clip)
            r_p = tl.load(tl.multiple_of(pm_ptr + offs, ALIGN))
            r_m = tl.load(tl.multiple_of(m_ptr + offs, ALIGN))
            r_v = tl.load(tl.multiple_of(v_ptr + offs, ALIGN))
            r_m = _fma(beta1, r_m, _mul(omb1, r_g))
            r_v = _fma(beta2, r_v, _mul(_mul(omb2, r_g), r_g))
            next_m = _div(r_m, bc1)
            next_v = _div(r_v, bc2)
            denom = _add(_sqrt(next_v), eps)
            q = _div(next_m, denom)
            update = _fma(decay, r_p, q)
            r_p = _fma(_mul(lr, -1.0), update, r_p)   # a true negation: Triton's -lr is 0 - lr (+0 for lr == 0)
            tl.store(tl.multiple_of(pm_ptr + offs, ALIGN), r_p)
            tl.store(tl.multiple_of(m_ptr + offs, ALIGN), r_m)
            tl.store(tl.multiple_of(v_ptr + offs, ALIGN), r_v)
            r_pb = r_p.to(tl.bfloat16)
            tl.store(tl.multiple_of(p_ptr + offs, ALIGN), r_pb)
            if AMAX:
                ub = r_pb.to(tl.int16, bitcast=True).to(tl.int32) & 0x7FFF
                ub = tl.where(ub > 0x7F80, 0, ub)
                amx = tl.maximum(amx, ub)
    if AMAX:
        slot = tl.load(slot_ptr + t)
        if slot >= 0:
            tl.atomic_max(amax_ptr + slot, tl.max(amx, axis=0) << 16)


@triton.jit
def _adamw_tail_kernel(addr_ptr, numel_ptr, tob_ptr, toff_ptr, clip_ptr,
                       lr, beta1, omb1, beta2, omb2, bc1, bc2, eps, decay, slot_ptr, amax_ptr,
                       BLOCK: tl.constexpr, ALIGN: tl.constexpr, AMAX: tl.constexpr):
    """One shard's ragged tail (fewer than BLOCK elements) per program."""
    pid = tl.program_id(0)
    t = tl.load(tob_ptr + pid)
    n = tl.load(numel_ptr + t)
    start = tl.load(toff_ptr + pid)
    g_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 0), tl.pointer_type(tl.bfloat16))
    pm_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 1), tl.pointer_type(tl.float32))
    m_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 2), tl.pointer_type(tl.float32))
    v_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 3), tl.pointer_type(tl.float32))
    p_ptr = tl.cast(tl.load(addr_ptr + t * 5 + 4), tl.pointer_type(tl.bfloat16))
    clip = tl.load(clip_ptr)

    offs = start + tl.arange(0, BLOCK)
    mask = offs < n
    r_g = _mul(tl.load(tl.multiple_of(g_ptr + offs, ALIGN), mask=mask, other=0.0).to(tl.float32), clip)
    r_p = tl.load(tl.multiple_of(pm_ptr + offs, ALIGN), mask=mask, other=0.0)
    r_m = tl.load(tl.multiple_of(m_ptr + offs, ALIGN), mask=mask, other=0.0)
    r_v = tl.load(tl.multiple_of(v_ptr + offs, ALIGN), mask=mask, other=0.0)
    r_m = _fma(beta1, r_m, _mul(omb1, r_g))
    r_v = _fma(beta2, r_v, _mul(_mul(omb2, r_g), r_g))
    next_m = _div(r_m, bc1)
    next_v = _div(r_v, bc2)
    denom = _add(_sqrt(next_v), eps)
    q = _div(next_m, denom)
    update = _fma(decay, r_p, q)
    r_p = _fma(_mul(lr, -1.0), update, r_p)   # a true negation: Triton's -lr is 0 - lr (+0 for lr == 0)
    tl.store(tl.multiple_of(pm_ptr + offs, ALIGN), r_p, mask=mask)
    tl.store(tl.multiple_of(m_ptr + offs, ALIGN), r_m, mask=mask)
    tl.store(tl.multiple_of(v_ptr + offs, ALIGN), r_v, mask=mask)
    r_pb = r_p.to(tl.bfloat16)
    tl.store(tl.multiple_of(p_ptr + offs, ALIGN), r_pb, mask=mask)
    if AMAX:
        ub = r_pb.to(tl.int16, bitcast=True).to(tl.int32) & 0x7FFF
        ub = tl.where(mask & (ub <= 0x7F80), ub, 0)
        slot = tl.load(slot_ptr + t)
        if slot >= 0:
            tl.atomic_max(amax_ptr + slot, tl.max(ub, axis=0) << 16)


def te_launch_table(numels):
    """TE's multi_tensor_apply bookkeeping for depth 1: for every chunk of every tensor, in
    order, the (launch, slot) it lands in.  Returns (tensor_of_chunk, chunk_index, launch*320+slot,
    blocks_per_launch)."""
    tob, cidx, dest, nblk = [], [], [], []
    loc_block = 0
    loc_tensor = 0
    launch = 0
    nt = len(numels)
    for t, n in enumerate(numels):
        loc_tensor += 1
        chunks = (n + _L2_CHUNK - 1) // _L2_CHUNK
        for c in range(chunks):
            tob.append(t); cidx.append(c); dest.append(launch * _L2_SLOTS + loc_block)
            loc_block += 1
            tensors_full = (loc_tensor == _L2_MAX_T and c == chunks - 1)
            blocks_full = (loc_block == _L2_SLOTS)
            last = (t == nt - 1 and c == chunks - 1)
            if tensors_full or blocks_full or last:
                nblk.append(loc_block)
                launch += 1
                loc_block = 0
                loc_tensor = 0 if c == chunks - 1 else 1
    return tob, cidx, dest, nblk


class FusedAdamWMaster:
    """One call = the gradient norm, the clip coefficient and the whole AdamW step over a list of
    shards.  The tables are rebuilt only when the list identity changes (the training loop's
    lists are static)."""

    def __init__(self, device="cuda"):
        self.device = device
        self._key = None
        self._lists = None
        self._addr = None
        self._numel = None
        self._bulk_tob = None
        self._bulk_sblk = None
        self._tail_tob = None
        self._tail_off = None
        self._grid_bulk = 0
        self._grid_tail = 0
        self._align = 1
        self._l2_kind = None
        self._l2_tob = None
        self._l2_c0 = None
        self._l2_nch = None
        self._l2_gidx = None
        self._l2_dest = None
        self._l2_nblk = None
        self._l2_partial = None
        self._l2_grid = 0
        self._l2_nlaunch = 0
        self._norm_addr = None
        self._norm_numel = None
        self._norm_aligned = None
        self._norm = torch.zeros(1, dtype=torch.float32, device=device)
        self._norm_key = None
        self._norm_lists = None
        self._norm_tensors = None

    def _prepare(self, grads, masters, ms, vs, params):
        lists = (grads, masters, ms, vs, params)
        cur = self._lists
        if cur is not None and all(a is b for a, b in zip(lists, cur)):
            return
        key = tuple([x.data_ptr() for x in grads] + [x.data_ptr() for x in masters])
        if key == self._key:
            self._lists = lists
            return
        n = [int(g.numel()) for g in grads]
        for tensors in lists:
            for x, k in zip(tensors, n):
                assert x.is_contiguous() and x.numel() == k, "every shard must be contiguous and the lists aligned"
        addrs = []
        for g, pm, m, v, p in zip(grads, masters, ms, vs, params):
            assert g.dtype == torch.bfloat16 and p.dtype == torch.bfloat16
            assert pm.dtype == torch.float32 and m.dtype == torch.float32 and v.dtype == torch.float32
            addrs += [g.data_ptr(), pm.data_ptr(), m.data_ptr(), v.data_ptr(), p.data_ptr()]

        # the step kernels' alignment contract: largest power of two dividing every pointer
        al = 64
        for a in addrs:
            while al > 1 and (a % al):
                al //= 2
        self._align = int(min(16, max(1, al // 4)))

        nt = len(n)
        full = [k // _BLOCK for k in n]
        chunks = [(f + _NBLK - 1) // _NBLK for f in full]
        bulk_tob = np.repeat(np.arange(nt, dtype=np.int32), chunks)
        if bulk_tob.size:
            bulk_sblk = np.concatenate([np.arange(c, dtype=np.int32) * _NBLK for c in chunks])
        else:
            bulk_sblk = np.zeros(0, dtype=np.int32)
        tail_tob = [i for i in range(nt) if n[i] % _BLOCK]
        tail_off = [full[i] * _BLOCK for i in tail_tob]

        dev = self.device
        self._addr = torch.tensor(addrs, dtype=torch.int64, device=dev)
        self._numel = torch.tensor(n, dtype=torch.int32, device=dev)
        self._bulk_tob = torch.from_numpy(np.ascontiguousarray(bulk_tob)).to(dev)
        self._bulk_sblk = torch.from_numpy(np.ascontiguousarray(bulk_sblk)).to(dev)
        self._tail_tob = torch.tensor(tail_tob, dtype=torch.int32, device=dev)
        self._tail_off = torch.tensor(tail_off, dtype=torch.int32, device=dev)
        self._grid_bulk = int(bulk_tob.size)
        self._grid_tail = len(tail_tob)
        self._key = key
        self._lists = lists

    def _prepare_norm(self, grads):
        """The norm's own tables (grads only: the norm is called before the step's lists exist).

        The training loop hands ``inputs["grads"]`` -- the SAME list object -- on every step, so
        the identity test is the fast path: without it this method rebuilds a 418-entry
        ``data_ptr()`` tuple and compares it on every call, ~40-100 us of Python sitting in front
        of the round's first kernel launch while the GPU is idle (the harness flushes the L2 and
        syncs between rounds).  A *different* list object holding the same shard objects is
        caught by the elementwise identity test, and the ``data_ptr`` key remains the last
        resort, so the tables are still rebuilt whenever a shard object or an address differs.

        The program table is a SINGLE list: every chunk of the round, tagged with the class its
        shape belongs to, so ``norm()`` launches one chunk grid plus the finish kernel.  The
        classes are appended in the order the three-grid version launched them (complete chunks,
        then tails, then the unaligned shards' chunks); the slots they write are disjoint, so the
        bytes are identical."""
        if self._norm_lists is grads:
            return
        cur = self._norm_tensors
        if cur is not None and len(cur) == len(grads) and all(a is b for a, b in zip(grads, cur)):
            self._norm_lists = grads
            return
        key = tuple(x.data_ptr() for x in grads)
        if key == self._norm_key:
            self._norm_lists = grads
            self._norm_tensors = tuple(grads)
            return
        n = [int(g.numel()) for g in grads]
        for g in grads:
            assert g.is_contiguous() and g.dtype == torch.bfloat16 and g.numel() > 0
        # TE's l2norm path per shard.  The shipped chain runs the norm on a FRESH fp32 copy of
        # each shard (`shard_model_grad.float()`, always 16-byte aligned), so its aligned code
        # path is taken exactly when numel % 4 == 0 -- whatever the bf16 shard's own offset is.
        # The candidate reads the bf16 shard in that path's ORDER; alignment of the loads is a
        # speed question, not a numerics one.
        aligned = [1 if k % 4 == 0 else 0 for k in n]
        b32 = [1 if (g.data_ptr() % 32) == 0 else 0 for g in grads]
        l2_tob, l2_cidx, l2_dest, l2_nblk = te_launch_table(n)
        nt = len(n)
        nchunks = [(k + _L2_CHUNK - 1) // _L2_CHUNK for k in n]
        first = np.zeros(nt, dtype=np.int64)
        if nt > 1:
            first[1:] = np.cumsum(nchunks)[:-1]

        kind, ktob, kc0, knch, kgidx = [], [], [], [], []
        # class 0 -- up to _L2_NCH consecutive COMPLETE chunks of a 32-byte aligned shard
        for t in range(nt):
            if b32[t]:
                nf = n[t] // _L2_CHUNK
                c = 0
                while c < nf:
                    k = min(_L2_NCH, nf - c)
                    kind.append(0); ktob.append(t); kc0.append(c); knch.append(k)
                    kgidx.append(int(first[t]) + c)
                    c += k
        # class 1 -- the single ragged TAIL chunk of a 32-byte aligned shard
        for t in range(nt):
            if b32[t]:
                nf = n[t] // _L2_CHUNK
                if nchunks[t] > nf:
                    kind.append(1); ktob.append(t); kc0.append(nf); knch.append(1)
                    kgidx.append(int(first[t]) + nf)
        # class 2 -- every chunk of a shard whose base is not 32-byte aligned
        for t in range(nt):
            if not b32[t]:
                c = 0
                while c < nchunks[t]:
                    k = min(_L2_NCH, nchunks[t] - c)
                    kind.append(2); ktob.append(t); kc0.append(c); knch.append(k)
                    kgidx.append(int(first[t]) + c)
                    c += k

        dev = self.device
        self._norm_addr = torch.tensor([g.data_ptr() for g in grads], dtype=torch.int64, device=dev)
        self._norm_numel = torch.tensor(n, dtype=torch.int32, device=dev)
        self._norm_aligned = torch.tensor(aligned, dtype=torch.int32, device=dev)
        self._l2_kind = torch.tensor(kind, dtype=torch.int32, device=dev)
        self._l2_tob = torch.tensor(ktob, dtype=torch.int32, device=dev)
        self._l2_c0 = torch.tensor(kc0, dtype=torch.int32, device=dev)
        self._l2_nch = torch.tensor(knch, dtype=torch.int32, device=dev)
        self._l2_gidx = torch.tensor(kgidx, dtype=torch.int32, device=dev)
        self._l2_dest = torch.from_numpy(np.asarray(l2_dest, dtype=np.int32)).to(dev)
        self._l2_grid = len(kind)
        self._l2_nblk = torch.tensor(l2_nblk, dtype=torch.int32, device=dev)
        self._l2_nlaunch = len(l2_nblk)
        self._l2_partial = torch.zeros(max(1, self._l2_nlaunch) * _L2_SLOTS, dtype=torch.float32, device=dev)
        self._norm_key = key
        self._norm_lists = grads
        self._norm_tensors = tuple(grads)

    def norm(self, grads):
        """TE's ``multi_tensor_l2norm`` over the bf16 shards: a 1-element fp32 device tensor with the
        kernel's bytes (the LOCAL norm; Megatron squares, all-reduces, roots and clips it)."""
        self._prepare_norm(grads)
        if self._l2_grid:
            _l2norm_chunk_kernel[(self._l2_grid,)](
                self._norm_addr, self._norm_numel, self._norm_aligned, self._l2_kind,
                self._l2_tob, self._l2_c0, self._l2_nch, self._l2_gidx,
                self._l2_dest, self._l2_partial,
                CHUNK=_L2_CHUNK, LANES=_L2_LANES, num_warps=4)
        _l2norm_finish_kernel[(1,)](
            self._l2_partial, self._l2_nblk, self._l2_nlaunch, self._norm,
            SLOTS=_L2_SLOTS, LANES=_L2_LANES, num_warps=4)
        return self._norm

    def step(self, grads, masters, ms, vs, params, clip_coeff, lr, beta1, beta2, eps, weight_decay, step,
             bias_correction=True, amax_buf=None, slots=None):
        """``clip_coeff``: a 1-element fp32 device tensor (Megatron's clamped clip coefficient).  x9step3: with
        ``amax_buf`` (fp32, zeroed by the caller) and ``slots`` (one int per shard, -1 = none) the kernels also fold
        each shard's max |bf16 param| into amax_buf[slot]."""
        self._prepare(grads, masters, ms, vs, params)
        amax_on = amax_buf is not None
        if amax_on:
            if getattr(self, "_slot_src", None) is not slots:
                assert len(slots) == len(grads) and amax_buf.dtype == torch.float32 and amax_buf.is_contiguous()
                self._slot = torch.tensor(list(slots), dtype=torch.int32, device=self.device)
                self._slot_src = slots
            slot_t, amax_t = self._slot, amax_buf.view(torch.int32)
        else:
            slot_t, amax_t = self._numel, self._numel   # unused (AMAX=False)
        b1, omb1, b2, omb2, bc1, bc2 = host_scalars(beta1, beta2, step, bias_correction)
        if hasattr(lr, "item"):
            lr = lr.item()
        lr = float(np.float32(lr))
        eps = float(np.float32(eps))
        decay = float(np.float32(weight_decay))
        align = self._align
        if self._grid_bulk:
            _adamw_bulk_kernel[(self._grid_bulk,)](
                self._addr, self._numel, self._bulk_tob, self._bulk_sblk, clip_coeff,
                lr, b1, omb1, b2, omb2, bc1, bc2, eps, decay, slot_t, amax_t,
                BLOCK=_BLOCK, NBLK=_NBLK, ALIGN=align, AMAX=amax_on, num_warps=_NUM_WARPS)
        if self._grid_tail:
            _adamw_tail_kernel[(self._grid_tail,)](
                self._addr, self._numel, self._tail_tob, self._tail_off, clip_coeff,
                lr, b1, omb1, b2, omb2, bc1, bc2, eps, decay, slot_t, amax_t,
                BLOCK=_BLOCK, ALIGN=align, AMAX=amax_on, num_warps=_NUM_WARPS)
