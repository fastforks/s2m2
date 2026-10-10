// Origin: exp 003 IGEV-general attempt/general-fused (core/cutlass_conv.cu) <- RAFT-Stereo exp 002 v8. S2M2 version:
// batch N, S2M2 epilogues (below), extra tile configs. CUTLASS v4.7.1 (cb42473) headers: experiments/002-raft-stereo/
// third_party/cutlass (or CUTLASS_DIR).
//
// Size-independent implicit-GEMM convolutions with fused epilogues (CUTLASS 2.x API, sm_80-style mma.sync with fp16
// accumulation), compiled ONCE ahead of time into prebuilt/libs2m2_cutlass_*.so -- no per-size compilation or tuning:
// the problem size is a runtime argument. Linear layers run as 1x1 convs (N=1, H=1, W=rows).
// The epilogue is an EpilogueWithVisitor: each thread's 8-channel accumulator vectors are finished from registers.
//   mode 0 (conv/linear + bias + act (+ residual)): v = act(acc + b[n]) (+ r[p*rs + n]) -> y[p*ys + n]; also -> y2[p*y2s + n]
//   mode 1 (ConvGRU z|r conv, K = 2*split): s = sigmoid(acc + b[n]); n < split: z[p*zs + n] = s;
//           else y[p*ys + n-split] = s * h[p*hs + n-split]                      (r*h into the [r*h | x] buffer)
//   mode 2 (ConvGRU q conv): q = tanh(acc + b[n]); hn = h + z (q - h), h = h[p*hs + n], z = z[p*zs + n] -> y[p*ys + n]
//           (y may alias h: each element is read and written by the same thread); also -> y2[p*y2s + n]
// act: 0 none, 1 relu, 3 tanh, 4 sigmoid, 6 exact GELU, 7 sigmoid clamped to [0.01, 0.99] (= s2m2_kernels.cu act_f).
// Numerics = s2m2_kernels.cu bias_act applied to an fp16 conv output (the accumulator is fp16 in both).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

extern "C" struct EpiArgs {
    int mode, act;
    const float* b;
    const __half* r; long long rs;
    __half* y; long long ys;
    __half* y2; long long y2s;
    const __half* h; long long hs;
    __half* z; long long zs;
    int split;
};

__device__ __forceinline__ float act_f(float v, int mode) {
    switch (mode) {
        case 1: return fmaxf(v, 0.f);
        case 3: return tanhf(v);
        case 4: return 1.f / (1.f + expf(-v));
        case 6: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
        case 7: return fminf(fmaxf(1.f / (1.f + expf(-v)), 0.01f), 0.99f);
        default: return v;
    }
}

// bias + activation (+ residual) of one 8-channel vector. Not inlined: the epilogue is unrolled over all fragments, and
// inlining the activation switch (erff / expf / tanhf) into every copy blows the instruction cache (1x1 convs +0.2 ms).
__device__ __noinline__ uint4 finish8(uint4 acc, const float* b, int act, uint4 rv, bool res) {
    const __half* ah = reinterpret_cast<const __half*>(&acc);
    const __half* rr = reinterpret_cast<const __half*>(&rv);
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float v = __half2float(ah[k]);
        if (b) v += b[k];
        v = act_f(v, act);
        if (res) v += __half2float(rr[k]);
        o[k] = __float2half(v);
    }
    return *reinterpret_cast<const uint4*>(o);
}

template <class OutIt>
struct S2Visitor {
    static int const kIterations = OutIt::kIterations;
    static int const kElementsPerAccess = OutIt::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "8 channels per access");
    using ElementAccumulator = Hf;
    using AccumulatorFragment = cutlass::Array<Hf, kElementsPerAccess>;
    using ThreadMap = typename OutIt::ThreadMap;
    struct SharedStorage {};

    EpiArgs const& a;
    OutIt it;
    int M, N;
    static int const kFrags = ThreadMap::Iterations::kCount;
    uint4 pr[kFrags], ph[kFrags];   // residual / h (and z) prefetched before the accumulator exchange

    CUTLASS_DEVICE S2Visitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                             int thread_idx, cutlass::MatrixCoord tb_offset)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()) {}

    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(i);
            long long p = co.row();
            int n = co.column();
            if (p < M && n < N) {
                if (a.mode == 0) {
                    if (a.r) pr[i] = __ldg(reinterpret_cast<const uint4*>(a.r + p * a.rs + n));
                } else if (a.mode == 1) {
                    if (n >= a.split) ph[i] = __ldg(reinterpret_cast<const uint4*>(a.h + p * a.hs + n - a.split));
                } else {
                    pr[i] = *reinterpret_cast<const uint4*>(a.z + p * a.zs + n);
                    ph[i] = *reinterpret_cast<const uint4*>(a.h + p * a.hs + n);
                }
            }
        }
    }
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void end_step(int) { ++it; }
    CUTLASS_DEVICE void end_epilogue() {}

    CUTLASS_DEVICE void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
        cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(frag_idx);
        long long p = co.row();
        int n = co.column();
        if (p >= M || n >= N) return;
        const __half* ah = reinterpret_cast<const __half*>(&acc);
        __align__(16) __half o[8];
        if (a.mode == 0) {
            uint4 out = finish8(*reinterpret_cast<const uint4*>(&acc), a.b ? a.b + n : nullptr, a.act, pr[frag_idx], a.r != nullptr);
            *reinterpret_cast<uint4*>(a.y + p * a.ys + n) = out;
            *reinterpret_cast<uint4*>(o) = out;
            if (a.y2) *reinterpret_cast<uint4*>(a.y2 + p * a.y2s + n) = *reinterpret_cast<const uint4*>(o);
        } else if (a.mode == 1) {
            if (n < a.split) {
#pragma unroll
                for (int k = 0; k < 8; ++k)
                    o[k] = __float2half(1.f / (1.f + expf(-(__half2float(ah[k]) + a.b[n + k]))));
                *reinterpret_cast<uint4*>(a.z + p * a.zs + n) = *reinterpret_cast<const uint4*>(o);
            } else {
                uint4 hv = ph[frag_idx];
                const __half* hh = reinterpret_cast<const __half*>(&hv);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    float r = 1.f / (1.f + expf(-(__half2float(ah[k]) + a.b[n + k])));
                    o[k] = __hmul(__float2half(r), hh[k]);
                }
                *reinterpret_cast<uint4*>(a.y + p * a.ys + n - a.split) = *reinterpret_cast<const uint4*>(o);
            }
        } else {
            uint4 zv = pr[frag_idx], hv = ph[frag_idx];
            const __half *zh = reinterpret_cast<const __half*>(&zv), *hh = reinterpret_cast<const __half*>(&hv);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                __half q = __float2half(tanhf(__half2float(ah[k]) + a.b[n + k]));
                // torch reference: addcmul(h, z, q - h) in fp16 tensors -> (q - h) rounded to fp16, then h + z*d
                float d = __half2float(__hsub(q, hh[k]));
                o[k] = __float2half(__half2float(hh[k]) + __half2float(zh[k]) * d);
            }
            *reinterpret_cast<uint4*>(a.y + p * a.ys + n) = *reinterpret_cast<const uint4*>(o);
            if (a.y2) *reinterpret_cast<uint4*>(a.y2 + p * a.y2s + n) = *reinterpret_cast<const uint4*>(o);
        }
    }
};

template <class TB, class WS, int Stages, int Swz = 1>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using Visitor = S2Visitor<OutIt>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        Visitor, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
};

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) fused_conv_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    cutlass::MatrixCoord extent(prm.problem_size.N * prm.problem_size.P * prm.problem_size.Q, prm.problem_size.K);
    typename F::Visitor v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run(const void* x, const void* w, int Nb, int H, int W, int C, int K, int R, int S, int ph, int pw, int stride,
               EpiArgs const& e, cudaStream_t s) {
    using Base = typename F::Base;
    int Ho = (H + 2 * ph - R) / stride + 1, Wo = (W + 2 * pw - S) / stride + 1;
    cutlass::conv::Conv2dProblemSize p({Nb, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {stride, stride}, {1, 1},
                                       {Nb, Ho, Wo, K}, cutlass::conv::Mode::kCrossCorrelation, 1);
    typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({Nb, H, W, C})}, {(Hf*)w, NHWC::packed({K, R, S, C})},
                                  {nullptr, NHWC::packed({Nb, Ho, Wo, K})}, {nullptr, NHWC::packed({Nb, Ho, Wo, K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(fused_conv_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr_set = true;
    }
    fused_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

// tile configurations (compiled once; the caller picks one per call by a fixed rule)
extern "C" int s2m2_conv(int cfg, const void* x, const void* w, int Nb, int H, int W, int C, int K, int R, int S, int ph,
                         int pw, int stride, const EpiArgs* e, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
#define ARGS x, w, Nb, H, W, C, K, R, S, ph, pw, stride, *e, s
    switch (cfg) {
        // Swizzle 8: tiles that share activation rows run together (n fastest) -> the activations are read from DRAM
        // once instead of once per 128-channel output tile (1x1 convs / linears: 2x faster).
        // Only the two configs the fixed rule uses are shipped (others screened in bench/s2m2_cutlass_probe.py:
        // 64x128x64/4st, 128x256, 256x128, 64x128, 128x64, 128x128x64, 4-stage, 2-stage, identity raster)
        case 0: return run<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 3, 8>>(ARGS);   // spatial convs (= cuDNN's tile)
        case 8: return run<Fused<GS<128, 128, 32>, GS<64, 32, 32>, 3, 8>>(ARGS);   // 8 warps: 1x1 convs / linears
    }
#undef ARGS
    return -1;
}
