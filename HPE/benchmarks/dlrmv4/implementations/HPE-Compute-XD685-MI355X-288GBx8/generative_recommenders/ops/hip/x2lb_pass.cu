// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef X2LB_INDEX_ONLY
#define X2LB_INDEX_ONLY 0
#endif
#if !X2LB_INDEX_ONLY
#include <hip/hip_runtime.h>
#endif
#include <stdint.h>

#include <utility>

#if !X2LB_INDEX_ONLY
namespace x2lb {
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}


extern "C" __device__ void x2lb_llvm_buffer_load_lds(x2lb::i32x4 rsrc, x2lb::lds_ptr_t lds, int size, int voffset,
                                                     int soffset, int offset, int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");
extern "C" __device__ void x2lb_llvm_buffer_store_v4i32(x2lb::i32x4 vdata, x2lb::i32x4 rsrc, int voffset, int soffset,
                                                        int aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");
extern "C" __device__ x2lb::i32x4 x2lb_llvm_buffer_load_v4i32(x2lb::i32x4 rsrc, int voffset, int soffset,
                                                             int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
extern "C" __device__ float x2lb_llvm_buffer_load_f32(x2lb::i32x4 rsrc, int voffset, int soffset,
                                                      int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");
#endif


#ifndef X2LB_LD_CP
#define X2LB_LD_CP 3
#endif
#ifndef X2LB_ST_CP
#define X2LB_ST_CP 3
#endif
#ifndef X2LB_DMA_CP
#define X2LB_DMA_CP 0
#endif


#ifndef X2LB_G3_KPERM
#define X2LB_G3_KPERM 0
#endif


#ifndef X2LB_G3_FMAEPI
#define X2LB_G3_FMAEPI 0
#endif


#ifndef X2LB_PROBE_NOFILL
#define X2LB_PROBE_NOFILL 0
#endif

namespace x2lb {


constexpr int KD = 512;
constexpr int NOUT = 256;
constexpr int BM = 64;
constexpr int NWAVE = 8;
constexpr int NTHREADS = 512;
constexpr int NSLOT = 4;
constexpr int ABYTES = 32768;
constexpr int RBASE = 2 * ABYTES;
constexpr int RPITCH = 528;
constexpr int RBYTES = BM * RPITCH;
constexpr int LDS_BYTES = RBASE + RBYTES;
constexpr uint32_t ROWB = 512;
constexpr uint32_t SLOT_STRIDE = 16 * ROWB;
constexpr int RED_FLOATS = 3 * 2 * NWAVE * NOUT;
static_assert(RED_FLOATS * 4 <= RBASE, "partial staging fits the (then free) A buffers");
static_assert(LDS_BYTES <= 163840, "LDS budget");


template <int EMIT>
constexpr int top_vmcnt() { return (EMIT ? 4 : 0) + NSLOT + NSLOT + 2; }


#if X2LB_INDEX_ONLY
#define X2LB_FN inline constexpr
#else
#define X2LB_FN __host__ __device__ __forceinline__ constexpr
#endif

X2LB_FN int static_tile(int b, int j, int G) { return b + j * G; }
X2LB_FN int tile_rows(int t, int M) { return M - t * BM < 0 ? 0 : (M - t * BM > BM ? BM : M - t * BM); }


X2LB_FN int a_dma_row(int w, int lane) { return 8 * w + (lane >> 3); }
X2LB_FN uint32_t a_dma_kswz(int w, int lane) {
    return (uint32_t)(16 * ((lane & 7) ^ ((a_dma_row(w, lane) >> 1) & 7)));
}
X2LB_FN uint32_t a_dma_voff(int w, int lane) { return (uint32_t)a_dma_row(w, lane) * KD + a_dma_kswz(w, lane); }
X2LB_FN uint32_t a_dma_lds(int buf, int w, int i) { return (uint32_t)(buf * ABYTES + i * 8192 + w * 1024); }
X2LB_FN uint32_t a_dma_soff(int i) { return (uint32_t)(i * 128); }


X2LB_FN bool zfill_lane(int rows, int w, int lane) {
    return !X2LB_PROBE_NOFILL && rows > 0 && rows < BM && a_dma_row(w, lane) >= rows;
}


X2LB_FN int rsrc_tile(int t, int rows) { return rows > 0 ? t : 0; }


X2LB_FN uint32_t frag_lp0(int lane) {
    return (uint32_t)(lane & 15) * 128u + (((uint32_t)(16 * (lane >> 4))) ^ ((uint32_t)((lane & 15) >> 1) << 4));
}
X2LB_FN uint32_t frag_lp1(int lane) {
    return (uint32_t)(lane & 15) * 128u + (((uint32_t)(16 * (lane >> 4) + 64)) ^ ((uint32_t)((lane & 15) >> 1) << 4));
}
X2LB_FN int frag_off(int buf, int k, int m) { return buf * ABYTES + k * 8192 + m * 2048; }


X2LB_FN int w_col(int w, int n, int lane) { return 32 * w + 8 * ((lane & 15) >> 2) + 4 * n + (lane & 3); }
X2LB_FN uint32_t w_voff(int w, int lane) { return (uint32_t)w_col(w, 0, lane) * KD + (uint32_t)(16 * (lane >> 4)); }
X2LB_FN uint32_t w_imm(int n, int k, int h) { return (uint32_t)(4 * n * KD + 128 * k + 64 * h); }

X2LB_FN int d_row(int m, int lane) { return 16 * m + (lane & 15); }
X2LB_FN int d_col(int w, int n, int i, int lane) { return 32 * w + 8 * (lane >> 4) + 4 * n + i; }


X2LB_FN uint32_t rw_base(int w, int lane) {
    return (uint32_t)(RBASE + (lane & 15) * RPITCH + 64 * w + 16 * (lane >> 4));
}
X2LB_FN int c_row(int w, int u, int lane) { return 16 * u + 2 * w + (lane >> 5); }
X2LB_FN int c_col0(int lane) { return 8 * (lane & 31); }
X2LB_FN uint32_t rr_base(int w, int lane) {
    return (uint32_t)(RBASE + (2 * w + (lane >> 5)) * RPITCH + 2 * c_col0(lane));
}

X2LB_FN uint32_t c_voff(int w, int u, int lane) { return (uint32_t)c_row(w, u, lane) * ROWB + 2u * (uint32_t)c_col0(lane); }

X2LB_FN uint32_t e_voff(int w, int m, int lane) {
    return (uint32_t)d_row(m, lane) * ROWB + 2u * (uint32_t)(32 * w + 8 * (lane >> 4));
}

X2LB_FN int acc_slot(int w, int lane) { return 2 * w + (lane >> 5); }
X2LB_FN int red_index(int o, int slot, int c) { return (o * 2 * NWAVE + slot) * NOUT + c; }

#if !X2LB_INDEX_ONLY


__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}


#define X2LB_FENCE() asm volatile("" ::: "memory")
#define X2LB_SB() __builtin_amdgcn_sched_barrier(0)
#define X2LB_BARRIER()                \
    do {                              \
        X2LB_FENCE();                 \
        __builtin_amdgcn_s_barrier(); \
        X2LB_FENCE();                 \
        X2LB_SB();                    \
    } while (0)


template <int IMM>
__device__ __forceinline__ i32x4 ds_read16(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x4 v;
    asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
template <int IMM>
__device__ __forceinline__ void ds_write16(uint32_t vaddr, i32x4 v) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    asm volatile("ds_write_b128 %0, %1 offset:%2" : : "v"(vaddr), "v"(v), "i"(IMM) : "memory");
}
template <int S>
__device__ __forceinline__ i32x8 frag(uint32_t lp0, uint32_t lp1) {
    i32x4 lo, hi;
    if constexpr (S < 65536) {
        lo = ds_read16<S>(lp0);
        hi = ds_read16<S>(lp1);
    } else {
        lo = ds_read16<S - 65536>(lp0 + 65536u);
        hi = ds_read16<S - 65536>(lp1 + 65536u);
    }
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}


constexpr int wenc(int vm, int lgkm) { return (vm & 15) | ((vm >> 4) << 14) | (7 << 4) | (lgkm << 8); }
template <int VM, int LGKM>
__device__ __forceinline__ void waitcnt() {
    static_assert(VM >= 0 && VM <= 63 && LGKM >= 0 && LGKM <= 15, "simm16 fields");
    X2LB_SB();
    X2LB_FENCE();
    __builtin_amdgcn_s_waitcnt(wenc(VM, LGKM));
    X2LB_FENCE();
    X2LB_SB();
}


template <int N>
__device__ __forceinline__ void wait_vm_lgkm0() { waitcnt<N, 0>(); }


template <bool FIRST>
__device__ __forceinline__ void mma2(f32x4 (&c)[2], const i32x8 (&b)[2], const i32x8& a) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        if constexpr (FIRST) {
            const f32x4 z = {0.f, 0.f, 0.f, 0.f};
            c[n] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[n], a, z, 0, 1, 0, 0, 0, 0);
        } else {
            c[n] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(b[n], a, c[n], 0, 1, 0, 0, 0, 0);
        }
    }
}


__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    x2lb_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, X2LB_DMA_CP);
}


__device__ __forceinline__ uint32_t pk_bf16(f32x2 v) {
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2));
}

__device__ __forceinline__ uint32_t epi2(f32x2 v, f32x2 s2, f32x2 z2) {
#if X2LB_G3_FMAEPI
    v = __builtin_elementwise_fma(v, s2, z2);
#else
    v = v * s2;
    v = v + z2;
#endif
    return pk_bf16(v);
}


__device__ __forceinline__ float bf_lo(uint32_t u) { return __builtin_bit_cast(float, u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __builtin_bit_cast(float, u & 0xffff0000u); }

__device__ __forceinline__ f32x2 bf2(int d) { return f32x2{bf_lo((uint32_t)d), bf_hi((uint32_t)d)}; }


template <int CTRL>
__device__ __forceinline__ float dpp_add(float v) {
    const int s = __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, true);
    return __builtin_bit_cast(float, s) + v;
}


template <int N, typename F>
__device__ __forceinline__ void static_for(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(
        std::make_integer_sequence<int, N>{});
}


__device__ __forceinline__ f32x2 fma2(f32x2 a, f32x2 b, f32x2 c) { return __builtin_elementwise_fma(a, b, c); }


__device__ __forceinline__ float hsum8(f32x2 p0, f32x2 p1, f32x2 p2, f32x2 p3) {
    const f32x2 h4a = p0 + p2;
    const f32x2 h4b = p1 + p3;
    const f32x2 h2 = h4a + h4b;
    return h2.x + h2.y;
}


__device__ __forceinline__ float row_tree32(float s) {
    s = dpp_add<0xB1>(s);
    s = dpp_add<0x4E>(s);
    s = dpp_add<0x141>(s);
    s = dpp_add<0x140>(s);
    const int o = __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, s), 0x401F);
    return s + __builtin_bit_cast(float, o);
}


struct Params {
    const uint8_t* dh8;
    const uint8_t* w8t;
    const float* sa;
    const float* sb;
    const uint16_t* h1;
    const float* mean1;
    const float* rstd1;
    const float* nw;
    const float* nb;
    uint16_t* dh1;
    uint16_t* da1;
    float* part;
    int M;
    int ntiles;
};


template <int S0, int EMIT>
__global__ __launch_bounds__(512) void x2lb_pass(const Params p) {
#pragma clang fp contract(off)
    static_assert(!(S0 && EMIT), "S0 has no da1 to emit");
    __shared__ __attribute__((aligned(2048))) uint8_t smem[LDS_BYTES];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int G = (int)gridDim.x;
    const int B = (int)blockIdx.x;
    const int M = p.M;
    const int ntiles = p.ntiles;


    float s = 0.f;
    if constexpr (!S0) {
        const float sav = p.sa[0], sbv = p.sb[0];
        s = sav * sbv;
        asm volatile("" : "+v"(s));
    }


    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    const f32x2 s2 = {s, s}, z2 = {0.f, 0.f};

    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);

    const uint32_t voffA = a_dma_voff(w, lane);
    const uint32_t ldsw = lds0 + a_dma_lds(0, w, 0);
    const uint32_t ldsz = ldsw + 16u * (uint32_t)lane;
    const uint32_t lp0 = lds0 + frag_lp0(lane);
    const uint32_t lp1 = lds0 + frag_lp1(lane);
    const uint32_t wR = lds0 + rw_base(w, lane);
    const uint32_t rR = lds0 + rr_base(w, lane);
    const uint32_t vC = c_voff(w, 0, lane);
    const uint32_t vE = e_voff(w, 0, lane);
    const bool hi_half = lane >= 32;


    i32x8 wf[2][4];
    if constexpr (!S0) {
        const i32x4 rW = make_rsrc(p.w8t, (uint32_t)(NOUT * KD));
        const uint32_t vw = w_voff(w, lane);
        static_for<2>([&](auto N_) {
            constexpr int n = decltype(N_)::value;
            static_for<4>([&](auto K_) {
                constexpr int k = decltype(K_)::value;
                const i32x4 lo = x2lb_llvm_buffer_load_v4i32(rW, (int)(vw + w_imm(n, k, 0)), 0, 0);
                const i32x4 hi = x2lb_llvm_buffer_load_v4i32(rW, (int)(vw + w_imm(n, k, 1)), 0, 0);
                wf[n][k] = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            });
        });
    }


    f32x2 wv[4], w2v[4], b2v[4];
    {
        const int c0 = c_col0(lane);
        const f32x4 wa = *reinterpret_cast<const f32x4*>(p.nw + c0);
        const f32x4 wb = *reinterpret_cast<const f32x4*>(p.nw + c0 + 4);
        const f32x4 ba = *reinterpret_cast<const f32x4*>(p.nb + c0);
        const f32x4 bb = *reinterpret_cast<const f32x4*>(p.nb + c0 + 4);
        const f32x2 nl2e = {__builtin_bit_cast(float, 0xBFB8AA3Bu), __builtin_bit_cast(float, 0xBFB8AA3Bu)};
        wv[0] = f32x2{wa.x, wa.y};
        wv[1] = f32x2{wa.z, wa.w};
        wv[2] = f32x2{wb.x, wb.y};
        wv[3] = f32x2{wb.z, wb.w};
        const f32x2 bv[4] = {f32x2{ba.x, ba.y}, f32x2{ba.z, ba.w}, f32x2{bb.x, bb.y}, f32x2{bb.z, bb.w}};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w2v[i] = wv[i] * nl2e;
            b2v[i] = bv[i] * nl2e;
        }
    }

    f32x2 adw[4], adb[4], adx[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        adw[i] = f32x2{0.f, 0.f};
        adb[i] = f32x2{0.f, 0.f};
        adx[i] = f32x2{0.f, 0.f};
    }


    i32x4 hx[NSLOT], dv[S0 ? NSLOT : 1];
    float mv, rv;


    auto rows_of = [&](int t) { return __builtin_amdgcn_readfirstlane(tile_rows(t, M)); };


    auto rs_rows = [&](const void* base, int t, int rows, uint32_t rowb) {
        return make_rsrc(reinterpret_cast<const uint8_t*>(base) + (size_t)rsrc_tile(t, rows) * BM * rowb,
                         (uint32_t)rows * rowb);
    };
    auto prefetch_slot = [&](int t, int rows, auto U_) {
        constexpr int u = decltype(U_)::value;
        hx[u] = x2lb_llvm_buffer_load_v4i32(rs_rows(p.h1, t, rows, ROWB), (int)(vC + (uint32_t)u * SLOT_STRIDE), 0,
                                            X2LB_LD_CP);
        if constexpr (S0)
            dv[u] = x2lb_llvm_buffer_load_v4i32(rs_rows(p.da1, t, rows, ROWB), (int)(vC + (uint32_t)u * SLOT_STRIDE),
                                                0, X2LB_LD_CP);
    };
    auto prefetch_stats = [&](int t, int rows) {
        mv = x2lb_llvm_buffer_load_f32(rs_rows(p.mean1, t, rows, 4u), lane * 4, 0, 0);
        rv = x2lb_llvm_buffer_load_f32(rs_rows(p.rstd1, t, rows, 4u), lane * 4, 0, 0);
    };


    auto issue_A = [&](int t, int rows, uint32_t bofs) {
        const i32x4 rA = rs_rows(p.dh8, t, rows, (uint32_t)KD);
        const uint32_t ldsb = ldsw + bofs;
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, ldsb + (uint32_t)(i * 8192), voffA, a_dma_soff(i));
        if (zfill_lane(rows, w, lane)) {
            const i32x4 zq = {0, 0, 0, 0};
            const uint32_t za = ldsz + bofs;
            static_for<4>([&](auto I_) {
                constexpr int i = decltype(I_)::value;
                ds_write16<i * 8192>(za, zq);
            });
        }
    };


    int t = static_tile(B, 0, G);
    {
        const int rows = rows_of(t);
        static_for<NSLOT>([&](auto U_) { prefetch_slot(t, rows, U_); });
        prefetch_stats(t, rows);
        if constexpr (!S0) issue_A(t, rows, 0u);
    }
    waitcnt<0, 15>();
    X2LB_SB();


    uint32_t bofs = 0;
    for (; t < ntiles; t += G) {
        const int tn = t + G;
        const int rows = rows_of(t), rows_n = rows_of(tn);

        if constexpr (!S0) wait_vm_lgkm0<top_vmcnt<EMIT>()>();
        else waitcnt<63, 0>();
        X2LB_BARRIER();

        if constexpr (!S0) issue_A(tn, rows_n, bofs ^ (uint32_t)ABYTES);
        X2LB_SB();
        const i32x4 rDH1 = rs_rows(p.dh1, t, rows, ROWB);

        if constexpr (!S0) {
            const i32x4 rDA1 = rs_rows(p.da1, t, rows, ROWB);
            const uint32_t la0 = lp0 + bofs, la1 = lp1 + bofs;
            i32x8 af[2][4];
            static_for<4>([&](auto K_) {
                constexpr int k = decltype(K_)::value;
                af[0][k] = frag<frag_off(0, k, 0)>(la0, la1);
            });
            asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(af[0][0]), "+v"(af[0][1]), "+v"(af[0][2]), "+v"(af[0][3]));
            X2LB_SB();
            static_for<4>([&](auto M_) {
                constexpr int m = decltype(M_)::value;
                constexpr int cur = m & 1;
                if constexpr (m < 3) {
                    static_for<4>([&](auto K_) {
                        constexpr int k = decltype(K_)::value;
                        af[cur ^ 1][k] = frag<frag_off(0, k, m + 1)>(la0, la1);
                    });
                }
                X2LB_SB();
                const i32x8 bw0[2] = {wf[0][0], wf[1][0]};
                const i32x8 bw1[2] = {wf[0][1], wf[1][1]};
                const i32x8 bw2[2] = {wf[0][2], wf[1][2]};
                const i32x8 bw3[2] = {wf[0][3], wf[1][3]};
                f32x4 acc[2];
                mma2<true>(acc, bw0, af[cur][0]);
                if constexpr (!X2LB_G3_KPERM) {
                    mma2<false>(acc, bw1, af[cur][1]);
                    mma2<false>(acc, bw2, af[cur][2]);
                } else {
                    mma2<false>(acc, bw2, af[cur][2]);
                    mma2<false>(acc, bw1, af[cur][1]);
                }
                mma2<false>(acc, bw3, af[cur][3]);

                i32x4 q;
                q.x = (int)epi2(__builtin_shufflevector(acc[0], acc[0], 0, 1), s2, z2);
                q.y = (int)epi2(__builtin_shufflevector(acc[0], acc[0], 2, 3), s2, z2);
                q.z = (int)epi2(__builtin_shufflevector(acc[1], acc[1], 0, 1), s2, z2);
                q.w = (int)epi2(__builtin_shufflevector(acc[1], acc[1], 2, 3), s2, z2);
                asm volatile("" : "+v"(q));
                ds_write16<m * 16 * RPITCH>(wR, q);
                if constexpr (EMIT)
                    x2lb_llvm_buffer_store_v4i32(q, rDA1, (int)(vE + (uint32_t)m * SLOT_STRIDE), 0, X2LB_ST_CP);
                X2LB_SB();
                if constexpr (m < 3) {
                    asm volatile("s_waitcnt lgkmcnt(0)"
                                 : "+v"(af[cur ^ 1][0]), "+v"(af[cur ^ 1][1]), "+v"(af[cur ^ 1][2]),
                                   "+v"(af[cur ^ 1][3]));
                    X2LB_SB();
                }
            });
        }

        waitcnt<63, 0>();
        X2LB_BARRIER();

        i32x4 dq[NSLOT];
        if constexpr (!S0) {
            static_for<NSLOT>([&](auto U_) {
                constexpr int u = decltype(U_)::value;
                dq[u] = ds_read16<u * 16 * RPITCH>(rR);
            });
            asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(dq[0]), "+v"(dq[1]), "+v"(dq[2]), "+v"(dq[3]));
        } else {
            static_for<NSLOT>([&](auto U_) { dq[decltype(U_)::value] = dv[decltype(U_)::value]; });
        }


        float smv[NSLOT][2], srv[NSLOT][2];
        static_for<NSLOT>([&](auto U_) {
            constexpr int u = decltype(U_)::value;
            const int rl0 = 16 * u + 2 * w;
            smv[u][0] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, mv), rl0));
            smv[u][1] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, mv), rl0 + 1));
            srv[u][0] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, rv), rl0));
            srv[u][1] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, rv), rl0 + 1));
        });
        X2LB_SB();
        prefetch_stats(tn, rows_n);
        X2LB_SB();
        static_for<NSLOT>([&](auto U_) {
            constexpr int u = decltype(U_)::value;


            asm volatile("" : "+v"(hx[u]), "+v"(dq[u]));

            const float mean = hi_half ? smv[u][1] : smv[u][0];
            const float rstd = hi_half ? srv[u][1] : srv[u][0];

            const f32x2 rs2 = {rstd, rstd};
            const float nmr = -(mean * rstd);
            const f32x2 nmr2 = {nmr, nmr};
            f32x2 xh[4], dy[4], sg[4], g[4], qv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const f32x2 x = bf2(hx[u][i]);
                dy[i] = bf2(dq[u][i]);
                xh[i] = fma2(x, rs2, nmr2);
                const f32x2 z = fma2(xh[i], w2v[i], b2v[i]);
                const float e0 = __builtin_amdgcn_exp2f(z.x);
                const float e1 = __builtin_amdgcn_exp2f(z.y);
                const float d0 = e0 + 1.0f;
                const float d1 = e1 + 1.0f;
                sg[i] = f32x2{__builtin_amdgcn_rcpf(d0), __builtin_amdgcn_rcpf(d1)};
                const f32x2 der = fma2(-sg[i], sg[i], sg[i]);
                const f32x2 pp = dy[i] * x;
                const f32x2 tt = pp * der;
                g[i] = wv[i] * tt;
                adw[i] = fma2(tt, xh[i], adw[i]);
                adb[i] = adb[i] + tt;
                qv[i] = xh[i] * g[i];
            }
            const float s1 = row_tree32(hsum8(qv[0], qv[1], qv[2], qv[3]));
            const float s2r = row_tree32(hsum8(g[0], g[1], g[2], g[3]));
            const float c1 = s1 * 0x1p-8f;
            const float c2 = s2r * 0x1p-8f;
            const float a1 = c1 * rstd;
            const float a2 = c2 * rstd;
            const f32x2 a12 = {a1, a1}, a22 = {a2, a2};
            i32x4 o;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const f32x2 inner = fma2(xh[i], a12, a22);
                const f32x2 v = fma2(g[i], rs2, -inner);
                const f32x2 dx = fma2(dy[i], sg[i], v);
                o[i] = (int)pk_bf16(dx);
                adx[i] = adx[i] + dx;
            }


            asm volatile("" : "+v"(adw[0]), "+v"(adw[1]), "+v"(adw[2]), "+v"(adw[3]));
            asm volatile("" : "+v"(adb[0]), "+v"(adb[1]), "+v"(adb[2]), "+v"(adb[3]));
            asm volatile("" : "+v"(adx[0]), "+v"(adx[1]), "+v"(adx[2]), "+v"(adx[3]));
            x2lb_llvm_buffer_store_v4i32(o, rDH1, (int)(vC + (uint32_t)u * SLOT_STRIDE), 0, X2LB_ST_CP);
            X2LB_SB();
            prefetch_slot(tn, rows_n, U_);
            X2LB_SB();
        });
        bofs ^= (uint32_t)ABYTES;
    }


    waitcnt<0, 0>();
    X2LB_BARRIER();
    {
        float* red = reinterpret_cast<float*>(smem);
        const int sl = acc_slot(w, lane), c0 = c_col0(lane);
        *reinterpret_cast<f32x4*>(red + red_index(0, sl, c0)) = f32x4{adw[0].x, adw[0].y, adw[1].x, adw[1].y};
        *reinterpret_cast<f32x4*>(red + red_index(0, sl, c0 + 4)) = f32x4{adw[2].x, adw[2].y, adw[3].x, adw[3].y};
        *reinterpret_cast<f32x4*>(red + red_index(1, sl, c0)) = f32x4{adb[0].x, adb[0].y, adb[1].x, adb[1].y};
        *reinterpret_cast<f32x4*>(red + red_index(1, sl, c0 + 4)) = f32x4{adb[2].x, adb[2].y, adb[3].x, adb[3].y};
        *reinterpret_cast<f32x4*>(red + red_index(2, sl, c0)) = f32x4{adx[0].x, adx[0].y, adx[1].x, adx[1].y};
        *reinterpret_cast<f32x4*>(red + red_index(2, sl, c0 + 4)) = f32x4{adx[2].x, adx[2].y, adx[3].x, adx[3].y};
        __syncthreads();
        for (int pi = (int)threadIdx.x; pi < 3 * NOUT; pi += NTHREADS) {
            const int o = pi / NOUT, c = pi % NOUT;
            float v[2 * NWAVE];
#pragma unroll
            for (int k = 0; k < 2 * NWAVE; ++k) v[k] = red[red_index(o, k, c)];

#pragma unroll
            for (int h = 1; h < 2 * NWAVE; h *= 2)
#pragma unroll
                for (int k = 0; k < 2 * NWAVE; k += 2 * h) v[k] = v[k] + v[k + h];
            p.part[((size_t)o * (size_t)G + (size_t)B) * NOUT + c] = v[0];
        }
    }
}


constexpr int FINB = 32;
__global__ __launch_bounds__(256) void x2lb_fin(const float* __restrict__ part, int G, float* __restrict__ out) {
#pragma clang fp contract(off)
    const int o = (int)blockIdx.x, c = (int)threadIdx.x;
    const float* pp = part + (size_t)o * (size_t)G * NOUT + c;
    float acc = 0.0f;
    int g = 0;
    for (; g + FINB <= G; g += FINB) {
        float v[FINB];
#pragma unroll
        for (int i = 0; i < FINB; ++i) v[i] = pp[(size_t)(g + i) * NOUT];
#pragma unroll
        for (int i = 0; i < FINB; ++i) acc = acc + v[i];
    }
    for (; g < G; ++g) acc = acc + pp[(size_t)g * NOUT];
    out[o * NOUT + c] = acc;
}
#endif

}
#if !X2LB_INDEX_ONLY


#ifndef X2LB_SRC_SHA256
#define X2LB_SRC_SHA256 "unknown"
#endif

#define X2LB_ABI_VERSION 2026101202
#define X2LB_STR2(x) #x
#define X2LB_STR(x) X2LB_STR2(x)


static constexpr long long X2LB_M_MAX = 1LL << 29;

extern "C" int x2lb_abi_version(void) { return X2LB_ABI_VERSION; }


extern "C" int x2lb_variant(void) {
    return (X2LB_G3_KPERM ? 1 : 0) | (X2LB_G3_FMAEPI ? 2 : 0) | (X2LB_PROBE_NOFILL ? 4 : 0);
}

extern "C" const char* x2lb_impl(void) {
    return "x2lb_pass v0.1 (S1 / S1+EMIT_DA1 / S0) BM64 8 waves persistent static tiles, resident W, lockstep; "
           "g3_kperm=" X2LB_STR(X2LB_G3_KPERM) " g3_fmaepi=" X2LB_STR(X2LB_G3_FMAEPI) " probe_nofill="
           X2LB_STR(X2LB_PROBE_NOFILL) " ld_cp=" X2LB_STR(X2LB_LD_CP) " st_cp=" X2LB_STR(X2LB_ST_CP)
           " dma_cp=" X2LB_STR(X2LB_DMA_CP) " src_sha256=" X2LB_SRC_SHA256;
}


extern "C" long long x2lb_plan(long long M, long long num_cu) {
    if (M < 1 || M > X2LB_M_MAX || num_cu < 1) return -(long long)hipErrorInvalidValue;
    const long long nt = (M + x2lb::BM - 1) / x2lb::BM;
    return nt < num_cu ? nt : num_cu;
}

extern "C" int x2lb_drain_last_error(void) { return (int)hipGetLastError(); }

static inline bool x2lb_a16(const void* q) { return q != nullptr && ((uintptr_t)q & 15u) == 0; }
static inline bool x2lb_a4(const void* q) { return q != nullptr && ((uintptr_t)q & 3u) == 0; }


template <int S0, int EMIT>
static int x2lb_launch_t(const void* dh8, const void* w8t, const void* sa, const void* sb, const void* h1,
                         const void* mean1, const void* rstd1, const void* nw, const void* nb, void* dh1, void* da1,
                         void* part, long long part_rows, long long M, long long grid, void* stream) {
    if (M < 1 || M > X2LB_M_MAX) return (int)hipErrorInvalidValue;
    const long long nt = (M + x2lb::BM - 1) / x2lb::BM;
    if (grid < 1 || grid > nt || grid > 0x7FFFFFFFLL) return (int)hipErrorInvalidValue;
    if (part_rows != grid) return (int)hipErrorInvalidValue;
    if (!(x2lb_a16(h1) && x2lb_a16(nw) && x2lb_a16(nb) && x2lb_a16(dh1) && x2lb_a16(part))) return (int)hipErrorInvalidValue;
    if (!(x2lb_a4(mean1) && x2lb_a4(rstd1))) return (int)hipErrorInvalidValue;
    if (!S0 && !(x2lb_a16(dh8) && x2lb_a16(w8t) && x2lb_a4(sa) && x2lb_a4(sb))) return (int)hipErrorInvalidValue;
    if ((S0 || EMIT) && !x2lb_a16(da1)) return (int)hipErrorInvalidValue;
    x2lb::Params prm{};
    prm.dh8 = (const uint8_t*)dh8;
    prm.w8t = (const uint8_t*)w8t;
    prm.sa = (const float*)sa;
    prm.sb = (const float*)sb;
    prm.h1 = (const uint16_t*)h1;
    prm.mean1 = (const float*)mean1;
    prm.rstd1 = (const float*)rstd1;
    prm.nw = (const float*)nw;
    prm.nb = (const float*)nb;
    prm.dh1 = (uint16_t*)dh1;
    prm.da1 = (uint16_t*)da1;
    prm.part = (float*)part;
    prm.M = (int)M;
    prm.ntiles = (int)nt;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((x2lb::x2lb_pass<S0, EMIT>), dim3((unsigned)grid), dim3(x2lb::NTHREADS), 0,
                       (hipStream_t)stream, prm);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

extern "C" int x2lb_launch(const void* dh8, const void* w8t, const void* sa, const void* sb, const void* h1,
                           const void* mean1, const void* rstd1, const void* nw, const void* nb, void* dh1, void* da1,
                           void* part, long long part_rows, long long M, long long grid, void* stream) {
    return x2lb_launch_t<0, 0>(dh8, w8t, sa, sb, h1, mean1, rstd1, nw, nb, dh1, da1, part, part_rows, M, grid, stream);
}
extern "C" int x2lb_launch_emit(const void* dh8, const void* w8t, const void* sa, const void* sb, const void* h1,
                                const void* mean1, const void* rstd1, const void* nw, const void* nb, void* dh1,
                                void* da1, void* part, long long part_rows, long long M, long long grid, void* stream) {
    return x2lb_launch_t<0, 1>(dh8, w8t, sa, sb, h1, mean1, rstd1, nw, nb, dh1, da1, part, part_rows, M, grid, stream);
}
extern "C" int x2lb_launch_s0(const void* dh8, const void* w8t, const void* sa, const void* sb, const void* h1,
                              const void* mean1, const void* rstd1, const void* nw, const void* nb, void* dh1, void* da1,
                              void* part, long long part_rows, long long M, long long grid, void* stream) {
    return x2lb_launch_t<1, 0>(dh8, w8t, sa, sb, h1, mean1, rstd1, nw, nb, dh1, da1, part, part_rows, M, grid, stream);
}


extern "C" int x2lb_fin_launch(const void* part, long long part_rows, void* out, void* stream) {
    if (part_rows < 1 || part_rows > 0x7FFFFFFFLL || !x2lb_a16(part) || !x2lb_a16(out)) return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL(x2lb::x2lb_fin, dim3(3), dim3(x2lb::NOUT), 0, (hipStream_t)stream, (const float*)part,
                       (int)part_rows, (float*)out);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}
#endif
