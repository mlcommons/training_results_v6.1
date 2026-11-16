// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <utility>

namespace wgck3 {
typedef int i32x2 __attribute__((ext_vector_type(2)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) uint32_t* lds_ptr_t;
}

extern "C" __device__ void wgck3_llvm_buffer_load_lds(wgck3::i32x4 rsrc, wgck3::lds_ptr_t lds, int size, int voffset,
                                                      int soffset, int offset, int aux)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

namespace wgck3 {

constexpr int KX = 512;
constexpr int KS = 128;
constexpr int A_ST = 256 * KS;
constexpr int B_ST = KS * 256;
constexpr int B_OFF = 2 * A_ST;
constexpr int FMT_G = 0;
constexpr uint32_t OOR = 0x80000000u;


__device__ __forceinline__ i32x4 make_rsrc(const void* p, uint32_t nrec) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    i32x4 r;
    r.x = (int)(uint32_t)a;
    r.y = (int)(uint32_t)(a >> 32);
    r.z = (int)nrec;
    r.w = 0x00110000;
    return r;
}

#define WGCK3_FENCE() asm volatile("" ::: "memory")
#define WGCK3_SB() __builtin_amdgcn_sched_barrier(0)
#define WGCK3_BARRIER()               \
    do {                              \
        WGCK3_FENCE();                \
        __builtin_amdgcn_s_barrier(); \
        WGCK3_FENCE();                \
        WGCK3_SB();                   \
    } while (0)

template <int IMM>
__device__ __forceinline__ i32x4 ds_read16(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x4 v;
    asm volatile("ds_read_b128 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}
template <int IMM>
__device__ __forceinline__ i32x2 ds_tr8(uint32_t vaddr) {
    static_assert(IMM >= 0 && IMM < 65536, "ds offset is 16 bit");
    i32x2 v;
    asm volatile("ds_read_b64_tr_b8 %0, %1 offset:%2" : "=v"(v) : "v"(vaddr), "i"(IMM) : "memory");
    return v;
}

template <int S>
__device__ __forceinline__ i32x8 fragA(uint32_t lp0, uint32_t lp1) {
    const i32x4 lo = ds_read16<S>(lp0);
    const i32x4 hi = ds_read16<S>(lp1);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}


template <int S>
__device__ __forceinline__ i32x8 fragB(uint32_t q) {
    const i32x2 r0 = ds_tr8<S>(q);
    const i32x2 r1 = ds_tr8<S + 2048>(q);
    const i32x2 r2 = ds_tr8<S + 16384>(q);
    const i32x2 r3 = ds_tr8<S + 16384 + 2048>(q);
    const i32x4 lo = __builtin_shufflevector(r0, r1, 0, 1, 2, 3);
    const i32x4 hi = __builtin_shufflevector(r2, r3, 0, 1, 2, 3);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}
__device__ __forceinline__ void g2l(i32x4 rsrc, uint32_t lds_addr, uint32_t voff, uint32_t soff) {
    wgck3_llvm_buffer_load_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_addr, 16, (int)voff, (int)soff, 0, 0);
}

}


template <int N, int TNW, int NSPLIT>
__global__ __launch_bounds__(512, 2) void wgrad_ck_w3_kernel(const uint8_t* __restrict__ G,
                                                             const uint8_t* __restrict__ X, long long ldm,
                                                             float* __restrict__ part, int M) {
    using namespace wgck3;
    static_assert(TNW == 192, "the production W3 tile is 256 x 192");
    static_assert(N % TNW == 0, "N must be a multiple of the tile width");
    static_assert(NSPLIT % 8 == 0, "NSPLIT must be a multiple of 8 (XCDs)");
    constexpr int NCH = TNW / 16;
    constexpr int NFB = TNW / 64;
    constexpr int NTN = N / TNW;
    constexpr int NT = 2 * NTN;
    constexpr int CPX = NSPLIT / 8;
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;
    const int cohort = loc / NT, tile = loc % NT;
    const int split = xcd * CPX + cohort;
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * TNW;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);


    const uint32_t chunkA = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    const uint32_t voffA = (uint32_t)((8 * w + (lane >> 3)) * ldm) + chunkA * 16u;

    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const int chunkB = (lane & 15) ^ fB;
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        voffB[i] = chunkB < NCH ? (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + (uint32_t)chunkB * 16u : OOR;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;


    const int r16 = lane & 15, q4 = lane >> 4;
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t rowA = lds0 + 16384u * (uint32_t)(w >> 2) + (uint32_t)r16 * 128u;
    const uint32_t lpA0 = rowA + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lpA1 = rowA + (((uint32_t)(16 * q4 + 64)) ^ swz);
    uint32_t qB[NFB];
    {
        const int s = r16, g = q4;
        const int t0 = 16 * g + (s >> 1);
        const int f = (s >> 1) + 8 * (g & 1);
        const int cw = NFB * (w & 3);
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb)
            qB[fb] = lds0 + (uint32_t)B_OFF + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) +
                     ((uint32_t)((cw + fb) ^ f) << 4);
    }

    f32x4 acc[8][NFB];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < NFB; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};

    const uint8_t* Xt = X + (long long)kx0 * ldm;
    const uint32_t nrecA = (uint32_t)min((long long)256 * ldm, (long long)0x7fffffffLL);


    auto issue = [&](int k, uint32_t st) {
        const i32x4 rA = make_rsrc(Xt + (long long)k * KS, nrecA);
        uint32_t vA = voffA;
        if ((long long)(k + 1) * KS > ldm) {
            if ((long long)k * KS + 16 * (long long)chunkA >= ldm) vA = OOR;
        }
        const int vt = min(KS, M - k * KS);
        const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), vA, (uint32_t)(64 * i * ldm));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
    };

    uint32_t pA0 = lpA0, pA1 = lpA1;
    uint32_t q[NFB];
#pragma unroll
    for (int fb = 0; fb < NFB; ++fb) q[fb] = qB[fb];
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK3_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK3_SB();
        i32x8 bf[NFB];
        i32x8 a[2];
        auto mma = [&](int fa, int fb, const i32x8& av) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[fb], av, acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };

        bf[0] = fragB<0>(q[0]);
        a[0] = fragA<0>(pA0, pA1);
        bf[1] = fragB<0>(q[1]);
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        bf[2] = fragB<0>(q[2]);
        WGCK3_SB();
        mma(0, 0, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        a[1] = fragA<2048>(pA0, pA1);
        WGCK3_SB();
        mma(0, 1, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        WGCK3_SB();
        mma(0, 2, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        WGCK3_SB();
        [&]<int... FB>(std::integer_sequence<int, FB...>) {
            (([&] {
                 constexpr int FA = FB + 1;
                 if constexpr (FA < 7) a[(FA + 1) & 1] = fragA<(FA + 1) * 2048>(pA0, pA1);
                 WGCK3_SB();
#pragma unroll
                 for (int fb = 0; fb < NFB; ++fb) mma(FA, fb, a[FA & 1]);
                 WGCK3_SB();
                 asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                 WGCK3_SB();
             }()),
             ...);
        }(std::make_integer_sequence<int, 7>{});
        st ^= 0x8000u;
        pA0 ^= 0x8000u;
        pA1 ^= 0x8000u;
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb) q[fb] ^= 0x8000u;
    }


    float* P = part + ((long long)split * KX + kx0 + 128 * (w >> 2) + r16) * N + n0 + (TNW / 4) * (w & 3) + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < 8; ++fa)
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}


template <int N, int TNW, int NSPLIT>
__global__ __launch_bounds__(512, 2) void wgrad_ck_w3_t_kernel(const uint8_t* __restrict__ G,
                                                               const uint8_t* __restrict__ X,
                                                               float* __restrict__ part, int M) {
    using namespace wgck3;
    static_assert(TNW == 192, "the production W3 tile is 256 x 192");
    static_assert(N % TNW == 0, "N must be a multiple of the tile width");
    static_assert(NSPLIT % 8 == 0, "NSPLIT must be a multiple of 8 (XCDs)");
    constexpr int NCH = TNW / 16;
    constexpr int NFB = TNW / 32;
    constexpr int NFA = 4;
    constexpr int NTN = N / TNW;
    constexpr int NT = 2 * NTN;
    constexpr int CPX = NSPLIT / 8;
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;
    const int cohort = loc / NT, tile = loc % NT;
    const int split = xcd * CPX + cohort;
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * TNW;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);


    const uint32_t chunkA = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    const uint32_t voffA = (uint32_t)((8 * w + (lane >> 3)) * KS) + chunkA * 16u;

    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const int chunkB = (lane & 15) ^ fB;
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        voffB[i] = chunkB < NCH ? (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + (uint32_t)chunkB * 16u : OOR;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;


    const int r16 = lane & 15, q4 = lane >> 4;
    const int rb = w >> 1, cb = w & 1;
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t rowA = lds0 + 8192u * (uint32_t)rb + (uint32_t)r16 * 128u;
    const uint32_t lpA0 = rowA + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lpA1 = rowA + (((uint32_t)(16 * q4 + 64)) ^ swz);
    uint32_t q[NFB];
    {
        const int s = r16, g = q4;
        const int t0 = 16 * g + (s >> 1);
        const int f = (s >> 1) + 8 * (g & 1);
        const int cw = NFB * cb;
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb)
            q[fb] = lds0 + (uint32_t)B_OFF + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) +
                    ((uint32_t)((cw + fb) ^ f) << 4);
    }

    f32x4 acc[NFA][NFB];
#pragma unroll
    for (int i = 0; i < NFA; ++i)
#pragma unroll
        for (int j = 0; j < NFB; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};


    auto issue = [&](int k, uint32_t st) {

        const i32x4 rA = make_rsrc(X + (long long)k * (KX * KS) + (long long)kx0 * KS, (uint32_t)(256 * KS));
        const int vt = min(KS, M - k * KS);
        const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), voffA, (uint32_t)(64 * i * KS));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
    };

    uint32_t pA0 = lpA0, pA1 = lpA1;
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK3_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK3_SB();
        i32x8 a[NFA];
        i32x8 bf[2];
        auto mma = [&](int fa, int fb, const i32x8& bv) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bv, a[fa], acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };
        a[0] = fragA<0>(pA0, pA1);
        bf[0] = fragB<0>(q[0]);
        a[1] = fragA<2048>(pA0, pA1);
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        a[2] = fragA<4096>(pA0, pA1);
        WGCK3_SB();
        mma(0, 0, bf[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        a[3] = fragA<6144>(pA0, pA1);
        WGCK3_SB();
        mma(1, 0, bf[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        bf[1] = fragB<0>(q[1]);
        WGCK3_SB();
        mma(2, 0, bf[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        WGCK3_SB();
        mma(3, 0, bf[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        WGCK3_SB();
        [&]<int... FBM>(std::integer_sequence<int, FBM...>) {
            (([&] {
                 constexpr int FB = FBM + 1;
                 if constexpr (FB + 1 < NFB) bf[(FB + 1) & 1] = fragB<0>(q[FB + 1]);
                 WGCK3_SB();
#pragma unroll
                 for (int fa = 0; fa < NFA; ++fa) mma(fa, FB, bf[FB & 1]);
                 WGCK3_SB();
                 asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                 WGCK3_SB();
             }()),
             ...);
        }(std::make_integer_sequence<int, NFB - 1>{});
        st ^= 0x8000u;
        pA0 ^= 0x8000u;
        pA1 ^= 0x8000u;
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb) q[fb] ^= 0x8000u;
    }


    float* P = part + ((long long)split * KX + kx0 + 64 * rb + r16) * N + n0 + (TNW / 2) * cb + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < NFA; ++fa)
#pragma unroll
        for (int fb = 0; fb < NFB; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}


template <int N, int NSPLIT>
__global__ __launch_bounds__(256) void wgrad_ck_w3_reduce(const float* __restrict__ part, float* __restrict__ out,
                                                          const float* __restrict__ gs, const float* __restrict__ as) {
#pragma clang fp contract(off)
    using namespace wgck3;
    constexpr long long Q = (long long)KX * N / 4;
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    const f32x4* p = reinterpret_cast<const f32x4*>(part);
    f32x4 s = p[i];
#pragma unroll 8
    for (int k = 1; k < NSPLIT; ++k) s = s + p[k * Q + i];
    const float sc = gs[0] * as[0];
    reinterpret_cast<f32x4*>(out)[i] = s * sc;
}


template <int N, int NSPLIT, int MAP>
__global__ __launch_bounds__(512, 2) void wgrad_ck_w3_t256s_kernel(const uint8_t* __restrict__ G, const uint8_t* __restrict__ X,
                                                          float* __restrict__ part, int M) {
    using namespace wgck3;
    constexpr int NTN = N / 256;
    constexpr int NT = 2 * NTN;
    constexpr int FMT_G = 0;
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;
    int split, tile;
    if constexpr (MAP == 0) {
        static_assert(NSPLIT % 8 == 0, "MAP 0 needs NSPLIT % 8 == 0");
        constexpr int CPX = NSPLIT / 8;
        split = xcd * CPX + loc / NT;
        tile = loc % NT;
    } else {
        constexpr int NWG = NSPLIT * NT;
        constexpr int LPX = 32 / NT;
        static_assert(NWG <= 256 && NSPLIT >= 8 * LPX, "spill map");
        if (loc < LPX * NT) {
            split = xcd * LPX + loc / NT;
            tile = loc % NT;
        } else {
            int e = loc - LPX * NT;
            for (int x = 0; x < xcd; ++x) e += (NWG - x + 7) / 8 - LPX * NT;
            split = 8 * LPX + e / NT;
            if constexpr (MAP == 2) {


                const int tp = e % NT;
                tile = (tp & 1) * NTN + (tp >> 1);
            } else {
                tile = e % NT;
            }
        }
    }
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * 256;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);

    const uint32_t chunkA = (uint32_t)((lane & 7) ^ ((4 * (w & 1) + (lane >> 4)) & 7));
    const uint32_t voffA = (uint32_t)((8 * w + (lane >> 3)) * KS) + chunkA * 16u;
    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const uint32_t chunkB = (uint32_t)((lane & 15) ^ fB);
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) voffB[i] = (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + chunkB * 16u;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;

    const int r16 = lane & 15, q4 = lane >> 4;
    const uint32_t swz = (uint32_t)(r16 >> 1) << 4;
    const uint32_t rowA = lds0 + 16384u * (uint32_t)(w >> 2) + (uint32_t)r16 * 128u;
    const uint32_t lpA0 = rowA + (((uint32_t)(16 * q4)) ^ swz);
    const uint32_t lpA1 = rowA + (((uint32_t)(16 * q4 + 64)) ^ swz);
    uint32_t qB[4];
    {
        const int s = r16, g = q4;
        const int t0 = 16 * g + (s >> 1);
        const int f = (s >> 1) + 8 * (g & 1);
        const int cw = 4 * (w & 3);
        const uint32_t q = lds0 + (uint32_t)B_OFF + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) + ((uint32_t)(cw ^ f) << 4);
#pragma unroll
        for (int fb = 0; fb < 4; ++fb) qB[fb] = q ^ ((uint32_t)fb << 4);
    }

    f32x4 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};

    auto issue = [&](int k, uint32_t st) {
        const i32x4 rA = make_rsrc(X + (long long)k * (KX * KS) + (long long)kx0 * KS, (uint32_t)(256 * KS));
        const int vt = min(KS, M - k * KS);
        const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), voffA, (uint32_t)(64 * i * KS));
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
    };

    uint32_t pA0 = lpA0, pA1 = lpA1;
    uint32_t q0 = qB[0], q1 = qB[1], q2 = qB[2], q3 = qB[3];
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK3_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK3_SB();
        i32x8 bf[4];
        i32x8 a[2];
        auto mma = [&](int fa, int fb, const i32x8& av) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[fb], av, acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };
        bf[0] = fragB<0>(q0);
        a[0] = fragA<0>(pA0, pA1);
        bf[1] = fragB<0>(q1);
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        bf[2] = fragB<0>(q2);
        WGCK3_SB();
        mma(0, 0, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        bf[3] = fragB<0>(q3);
        WGCK3_SB();
        mma(0, 1, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
        a[1] = fragA<2048>(pA0, pA1);
        WGCK3_SB();
        mma(0, 2, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(2)" ::: "memory");
        WGCK3_SB();
        mma(0, 3, a[0]);
        WGCK3_SB();
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        WGCK3_SB();
        [&]<int... FB>(std::integer_sequence<int, FB...>) {
            (([&] {
                 constexpr int FA = FB + 1;
                 if constexpr (FA < 7) a[(FA + 1) & 1] = fragA<(FA + 1) * 2048>(pA0, pA1);
                 WGCK3_SB();
#pragma unroll
                 for (int fb = 0; fb < 4; ++fb) mma(FA, fb, a[FA & 1]);
                 WGCK3_SB();
                 asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                 WGCK3_SB();
             }()),
             ...);
        }(std::make_integer_sequence<int, 7>{});
        st ^= 0x8000u;
        pA0 ^= 0x8000u;
        pA1 ^= 0x8000u;
        q0 ^= 0x8000u;
        q1 ^= 0x8000u;
        q2 ^= 0x8000u;
        q3 ^= 0x8000u;
    }

    float* P = part + ((long long)split * KX + kx0 + 128 * (w >> 2) + r16) * N + n0 + 64 * (w & 3) + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < 8; ++fa)
#pragma unroll
        for (int fb = 0; fb < 4; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}


static constexpr long long WGCK3_LDM_MAX = 0x7fffffffLL / 256;

template <int N, int TNW, int NSPLIT>
static int wgrad_ck_w3_launch(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                              const float* as, int M, void* stream) {
    if (M <= 0 || (ldm & 15) != 0 || ldm < ((M + 15) / 16) * 16LL || ldm > WGCK3_LDM_MAX)
        return (int)hipErrorInvalidValue;
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    constexpr int NT = 2 * (N / TNW);


    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_w3_kernel<N, TNW, NSPLIT>), dim3(NSPLIT * NT), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, ldm, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_w3_reduce<N, NSPLIT>), dim3((wgck3::KX * N / 4) / 256), dim3(256), 0,
                       (hipStream_t)stream, (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}


extern "C" int wgrad_ck_w3(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                           const float* as, int M, void* stream) {
    return wgrad_ck_w3_launch<1536, 192, 16>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_nsplit(void) { return 16; }


template <int N, int TNW, int NSPLIT>
static int wgrad_ck_w3_t_launch(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                const float* as, int M, void* stream) {
    if (M <= 0 || ldm != (long long)wgck3::KX * wgck3::KS) return (int)hipErrorInvalidValue;
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    constexpr int NT = 2 * (N / TNW);
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_w3_t_kernel<N, TNW, NSPLIT>), dim3(NSPLIT * NT), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_w3_reduce<N, NSPLIT>), dim3((wgck3::KX * N / 4) / 256), dim3(256), 0,
                       (hipStream_t)stream, (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}


extern "C" int wgrad_ck_w3_t(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                             const float* as, int M, void* stream) {
    return wgrad_ck_w3_t_launch<1536, 192, 16>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t_nsplit(void) { return 16; }


extern "C" int wgrad_ck_w3_drain_last_error(void) { return (int)hipGetLastError(); }

template <int MAP>
static int wgrad_ck_w3_t256s_launch(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                    const float* as, int M, void* stream) {
    constexpr int N = 1536, NS = 21;
    if (M <= 0 || ldm != (long long)wgck3::KX * wgck3::KS) return (int)hipErrorInvalidValue;
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_w3_t256s_kernel<N, NS, MAP>), dim3(NS * 12), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_w3_reduce<N, NS>), dim3((wgck3::KX * N / 4) / 256), dim3(256), 0, (hipStream_t)stream,
                       (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}

extern "C" int wgrad_ck_w3_t256s(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                 const float* as, int M, void* stream) {
    return wgrad_ck_w3_t256s_launch<2>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t256s_nsplit(void) { return 21; }
extern "C" int wgrad_ck_w3_t256s_m1(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                    const float* as, int M, void* stream) {
    return wgrad_ck_w3_t256s_launch<1>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t256s_m1_nsplit(void) { return 21; }


template <int N, int NSPLIT, int MAP, int PF>
__global__ __launch_bounds__(512, 2) void wgrad_ck_w3_t256s_xt2_kernel(const uint8_t* __restrict__ G,
                                                                       const uint8_t* __restrict__ X,
                                                                       float* __restrict__ part, int M) {
    using namespace wgck3;
    static_assert(PF == 1 || PF == 2, "PF: A-fragment prefetch depth 1 or 2");
    constexpr int NTN = N / 256;
    constexpr int NT = 2 * NTN;
    constexpr int FMT_G = 0;
    __shared__ __attribute__((aligned(4096))) uint8_t smem[2 * A_ST + 2 * B_ST];

    const int lane = (int)(threadIdx.x & 63);
    const int w = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 6));
    const int b = (int)blockIdx.x;
    const int xcd = b & 7, loc = b >> 3;

    int split, tile;
    if constexpr (MAP == 0) {
        static_assert(NSPLIT % 8 == 0, "MAP 0 needs NSPLIT % 8 == 0");
        constexpr int CPX = NSPLIT / 8;
        split = xcd * CPX + loc / NT;
        tile = loc % NT;
    } else {
        constexpr int NWG = NSPLIT * NT;
        constexpr int LPX = 32 / NT;
        static_assert(NWG <= 256 && NSPLIT >= 8 * LPX, "spill map");
        if (loc < LPX * NT) {
            split = xcd * LPX + loc / NT;
            tile = loc % NT;
        } else {
            int e = loc - LPX * NT;
            for (int x = 0; x < xcd; ++x) e += (NWG - x + 7) / 8 - LPX * NT;
            split = 8 * LPX + e / NT;
            if constexpr (MAP == 2) {
                const int tp = e % NT;
                tile = (tp & 1) * NTN + (tp >> 1);
            } else {
                tile = e % NT;
            }
        }
    }
    const int kx0 = (tile / NTN) * 256, n0 = (tile % NTN) * 256;
    const int KT = (M + KS - 1) / KS;
    const int kb = (int)(((long long)KT * split) / NSPLIT);
    const int ke = (int)(((long long)KT * (split + 1)) / NSPLIT);
    const uint32_t lds0 = (uint32_t)reinterpret_cast<uintptr_t>(&smem[0]);


    const int fB = 4 * (w & 1) + (lane >> 4) + 8 * ((w >> 2) & 1);
    const uint32_t chunkB = (uint32_t)((lane & 15) ^ fB);
    uint32_t voffB[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) voffB[i] = (uint32_t)((32 * i + 4 * w + (lane >> 4)) * N) + chunkB * 16u;
    const uint32_t ldsA_w = lds0 + 1024u * (uint32_t)w;
    const uint32_t ldsB_w = lds0 + (uint32_t)B_OFF + 1024u * (uint32_t)w;


    const uint32_t voffA2 = (uint32_t)((4 * w + (lane >> 4)) * KX) + chunkB * 16u;


    const int r16 = lane & 15, q4 = lane >> 4;
    uint32_t qB[4];
    uint32_t pX;
    {
        const int s = r16, g = q4;
        const int t0 = 16 * g + (s >> 1);
        const int f = (s >> 1) + 8 * (g & 1);
        const int cw = 4 * (w & 3);
        const uint32_t q = lds0 + (uint32_t)B_OFF + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) + ((uint32_t)(cw ^ f) << 4);
#pragma unroll
        for (int fb = 0; fb < 4; ++fb) qB[fb] = q ^ ((uint32_t)fb << 4);
        const int ca0 = 8 * (w >> 2);
        pX = lds0 + (uint32_t)t0 * 256u + 8u * (uint32_t)(s & 1) + ((uint32_t)(ca0 ^ f) << 4);
    }

    f32x4 acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};


    auto issue = [&](int k, uint32_t st) {


        const int vt = min(KS, M - k * KS);
        const i32x4 rB = make_rsrc(G + (long long)k * KS * N + n0, (uint32_t)(vt * N));
#pragma unroll
        for (int i = 0; i < 4; ++i) {


            const int vi = vt - 32 * i;
            const uint32_t nrA = (uint32_t)__builtin_amdgcn_readfirstlane((vi > 0 ? vi : 0) * KX);
            const i32x4 rA = make_rsrc(X + ((long long)k * KS + 32 * i) * KX + kx0, nrA);
            g2l(rA, (ldsA_w ^ st) + (uint32_t)(8192 * i), voffA2, 0u);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) g2l(rB, (ldsB_w ^ st) + (uint32_t)(8192 * i), voffB[i], 0u);
    };

    uint32_t q0 = qB[0], q1 = qB[1], q2 = qB[2], q3 = qB[3];
    uint32_t st = 0;
    if (kb < ke) issue(kb, 0u);
    for (int k = kb; k < ke; ++k) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        WGCK3_BARRIER();
        if (k + 1 < ke) issue(k + 1, st ^ 0x8000u);
        WGCK3_SB();
        i32x8 bf[4];
        auto mma = [&](int fa, int fb, const i32x8& av) {
            acc[fa][fb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(bf[fb], av, acc[fa][fb], FMT_G, 0, 0, 0, 0, 0);
        };
        if constexpr (PF == 1) {


            i32x8 a[2];
            bf[0] = fragB<0>(q0);
            a[0] = fragB<0>(pX);
            bf[1] = fragB<0>(q1);
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            bf[2] = fragB<0>(q2);
            WGCK3_SB();
            mma(0, 0, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            bf[3] = fragB<0>(q3);
            WGCK3_SB();
            mma(0, 1, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            a[1] = fragB<0>(pX ^ 16u);
            WGCK3_SB();
            mma(0, 2, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            WGCK3_SB();
            mma(0, 3, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            WGCK3_SB();
            [&]<int... FB>(std::integer_sequence<int, FB...>) {
                (([&] {
                     constexpr int FA = FB + 1;
                     if constexpr (FA < 7) a[(FA + 1) & 1] = fragB<0>(pX ^ ((uint32_t)(FA + 1) << 4));
                     WGCK3_SB();
#pragma unroll
                     for (int fb = 0; fb < 4; ++fb) mma(FA, fb, a[FA & 1]);
                     WGCK3_SB();
                     asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                     WGCK3_SB();
                 }()),
                 ...);
            }(std::make_integer_sequence<int, 7>{});
        } else {


            i32x8 a[3];
            bf[0] = fragB<0>(q0);
            a[0] = fragB<0>(pX);
            bf[1] = fragB<0>(q1);
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            bf[2] = fragB<0>(q2);
            WGCK3_SB();
            mma(0, 0, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            bf[3] = fragB<0>(q3);
            WGCK3_SB();
            mma(0, 1, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            a[1] = fragB<0>(pX ^ 16u);
            WGCK3_SB();
            mma(0, 2, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            a[2] = fragB<0>(pX ^ 32u);
            WGCK3_SB();
            mma(0, 3, a[0]);
            WGCK3_SB();
            asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
            WGCK3_SB();
            [&]<int... FB>(std::integer_sequence<int, FB...>) {
                (([&] {
                     constexpr int FA = FB + 1;
                     if constexpr (FA + 2 < 8) a[(FA + 2) % 3] = fragB<0>(pX ^ ((uint32_t)(FA + 2) << 4));
                     WGCK3_SB();
#pragma unroll
                     for (int fb = 0; fb < 4; ++fb) mma(FA, fb, a[FA % 3]);
                     WGCK3_SB();
                     if constexpr (FA + 2 < 8) {
                         asm volatile("s_waitcnt lgkmcnt(4)" ::: "memory");
                     } else {
                         asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
                     }
                     WGCK3_SB();
                 }()),
                 ...);
            }(std::make_integer_sequence<int, 7>{});
        }
        st ^= 0x8000u;
        pX ^= 0x8000u;
        q0 ^= 0x8000u;
        q1 ^= 0x8000u;
        q2 ^= 0x8000u;
        q3 ^= 0x8000u;
    }


    float* P = part + ((long long)split * KX + kx0 + 128 * (w >> 2) + r16) * N + n0 + 64 * (w & 3) + 4 * q4;
#pragma unroll
    for (int fa = 0; fa < 8; ++fa)
#pragma unroll
        for (int fb = 0; fb < 4; ++fb)
            *reinterpret_cast<f32x4*>(P + (long long)(16 * fa) * N + 16 * fb) = acc[fa][fb];
}

template <int MAP, int PF>
static int wgrad_ck_w3_t256s_xt2_launch(const void* G, const void* X, long long ldm, void* part, void* out,
                                        const float* gs, const float* as, int M, void* stream) {
    constexpr int N = 1536, NS = 21;


    if (M <= 0 || ldm != (long long)wgck3::KX) return (int)hipErrorInvalidValue;
    if (((uintptr_t)G | (uintptr_t)X | (uintptr_t)part | (uintptr_t)out) & 15) return (int)hipErrorInvalidValue;
    if (gs == nullptr || as == nullptr) return (int)hipErrorInvalidValue;
    const hipError_t stale = hipGetLastError();
    hipLaunchKernelGGL((wgrad_ck_w3_t256s_xt2_kernel<N, NS, MAP, PF>), dim3(NS * 12), dim3(512), 0, (hipStream_t)stream,
                       (const uint8_t*)G, (const uint8_t*)X, (float*)part, M);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    hipLaunchKernelGGL((wgrad_ck_w3_reduce<N, NS>), dim3((wgck3::KX * N / 4) / 256), dim3(256), 0, (hipStream_t)stream,
                       (const float*)part, (float*)out, gs, as);
    e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return stale == hipSuccess ? 0 : -(int)stale;
}


extern "C" int wgrad_ck_w3_t256s_xt2(const void* G, const void* X, long long ldm, void* part, void* out, const float* gs,
                                     const float* as, int M, void* stream) {
    return wgrad_ck_w3_t256s_xt2_launch<2, 1>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t256s_xt2_nsplit(void) { return 21; }

extern "C" int wgrad_ck_w3_t256s_xt2p2(const void* G, const void* X, long long ldm, void* part, void* out,
                                       const float* gs, const float* as, int M, void* stream) {
    return wgrad_ck_w3_t256s_xt2_launch<2, 2>(G, X, ldm, part, out, gs, as, M, stream);
}
extern "C" int wgrad_ck_w3_t256s_xt2p2_nsplit(void) { return 21; }


__global__ __launch_bounds__(512) void wgck3_lds_poison_kernel(uint32_t pattern) {
    __shared__ __attribute__((aligned(16))) uint32_t pz[40960];
    const uint32_t base = (uint32_t)reinterpret_cast<uintptr_t>(&pz[0]);
    for (int i = (int)threadIdx.x; i < 40960; i += 512)
        asm volatile("ds_write_b32 %0, %1" ::"v"(base + 4u * (uint32_t)i), "v"(pattern) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}
__global__ __launch_bounds__(512) void wgck3_lds_probe_kernel(uint32_t pattern, unsigned int* hits) {
    __shared__ __attribute__((aligned(16))) uint32_t pz[40960];
    const uint32_t base = (uint32_t)reinterpret_cast<uintptr_t>(&pz[0]);
    unsigned int n = 0;
    for (int i = (int)threadIdx.x; i < 40960; i += 512) {
        uint32_t v;
        asm volatile("ds_read_b32 %0, %1\n\ts_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(base + 4u * (uint32_t)i) : "memory");
        n += (v == pattern) ? 1u : 0u;
    }
    atomicAdd(hits, n);
}
extern "C" int wgck3_lds_poison(unsigned int pattern, int nwg, void* stream) {
    const hipError_t stale = hipGetLastError();
    (void)stale;
    hipLaunchKernelGGL(wgck3_lds_poison_kernel, dim3(nwg), dim3(512), 0, (hipStream_t)stream, (uint32_t)pattern);
    return (int)hipGetLastError();
}
extern "C" int wgck3_lds_probe(unsigned int pattern, int nwg, void* hits, void* stream) {
    const hipError_t stale = hipGetLastError();
    (void)stale;
    hipLaunchKernelGGL(wgck3_lds_probe_kernel, dim3(nwg), dim3(512), 0, (hipStream_t)stream, (uint32_t)pattern,
                       (unsigned int*)hits);
    return (int)hipGetLastError();
}
