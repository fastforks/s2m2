// Small size-independent CUDA kernels for S2M2 inference (NHWC fp16 rows [P, C]). Compiled once to a cubin
// (kernels.py), launched on torch's current stream. Pattern from IGEV-general / RAFT-Stereo exp 002.
#include <cuda_fp16.h>
#include <stdint.h>

#define GRID_STRIDE(i, n) for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

__device__ __forceinline__ float act_f(float v, int mode) {
    switch (mode) {
        case 1: return fmaxf(v, 0.f);                                     // relu
        case 3: return tanhf(v);
        case 4: return 1.f / (1.f + expf(-v));                            // sigmoid
        case 6: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); // exact GELU (nn.GELU default)
        case 7: return fminf(fmaxf(1.f / (1.f + expf(-v)), 0.01f), 0.99f); // sigmoid + clamp (FeatureFusion gate)
        default: return v;
    }
}

// y[p, c] = act(y[p, c] + b[c]) (+ r[p, c])   in place; b / r may be null. r is added after the activation.
extern "C" __global__ void bias_act(__half* y, const float* __restrict__ b, const __half* __restrict__ r,
                                    long long P, int C, int mode) {
    GRID_STRIDE(i, P * C) {
        int c = (int)(i % C);
        float v = __half2float(y[i]);
        if (b) v += b[c];
        v = act_f(v, mode);
        if (r) v += __half2float(r[i]);
        y[i] = __float2half(v);
    }
}

// 8 channels per thread, 16-byte accesses (C multiple of 8, 16-byte aligned pointers)
extern "C" __global__ void bias_act8(__half* y, const float* __restrict__ b, const __half* __restrict__ r,
                                     long long P, int C, int mode) {
    const int CV = C / 8;
    GRID_STRIDE(i, P * CV) {
        int c0 = (int)(i % CV) * 8;
        uint4 raw = reinterpret_cast<uint4*>(y)[i];
        __half* hv = reinterpret_cast<__half*>(&raw);
        uint4 rraw;
        __half* rv = reinterpret_cast<__half*>(&rraw);
        if (r) rraw = reinterpret_cast<const uint4*>(r)[i];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(hv[k]);
            if (b) v += b[c0 + k];
            v = act_f(v, mode);
            if (r) v += __half2float(rv[k]);
            hv[k] = __float2half(v);
        }
        reinterpret_cast<uint4*>(y)[i] = raw;
    }
}

// FeatureFusion output: y = (y + b) + z1 + w * (z0 - z1)   (all [P, C] dense, C multiple of 8)
extern "C" __global__ void gate_mix8(__half* y, const float* __restrict__ b, const __half* __restrict__ w,
                                     const __half* __restrict__ z0, const __half* __restrict__ z1, long long P, int C) {
    const int CV = C / 8;
    GRID_STRIDE(i, P * CV) {
        int c0 = (int)(i % CV) * 8;
        uint4 ry = reinterpret_cast<uint4*>(y)[i];
        uint4 rw = reinterpret_cast<const uint4*>(w)[i];
        uint4 r0 = reinterpret_cast<const uint4*>(z0)[i];
        uint4 r1 = reinterpret_cast<const uint4*>(z1)[i];
        __half* hy = reinterpret_cast<__half*>(&ry);
        const __half* hw = reinterpret_cast<const __half*>(&rw);
        const __half* h0 = reinterpret_cast<const __half*>(&r0);
        const __half* h1 = reinterpret_cast<const __half*>(&r1);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float a = __half2float(h0[k]), c = __half2float(h1[k]);
            float v = __half2float(hy[k]) + (b ? b[c0 + k] : 0.f);
            hy[k] = __float2half(v + c + __half2float(hw[k]) * (a - c));
        }
        reinterpret_cast<uint4*>(y)[i] = ry;
    }
}

// ------------------------------------------------------------------------------------------------ optimal transport
// Sinkhorn (log domain, as upstream DispInit) directly on the fp16 correlation cv [rows, w, w] (row r = image row,
// i = left pixel, j = right pixel); the dustbin row / column (index w) is implicit (score 0), the positivity mask
// (j > i -> -1e4) is applied on the fly. All math fp32 (== upstream's a = cv.float(); masked_fill; pad), no
// [rows, w+1, w+1] fp32 tensors. lm_a = log marginal of a pixel, lm_d = of the dustbin.
__device__ __forceinline__ float ot_a(const __half* __restrict__ row, int i, int j, int w, int pos) {
    if (i == w || j == w) return 0.f;
    if (pos && j > i) return -1e4f;
    return __half2float(row[j]);
}

__device__ __forceinline__ void lse_add(float& m, float& s, float x) {
    if (x > m) { s = s * expf(m - x) + 1.f; m = x; } else { s += expf(x - m); }
}

__device__ __forceinline__ void lse_merge(float& m, float& s, float m2, float s2) {
    if (m2 > m) { s = s * expf(m - m2) + s2; m = m2; } else if (s2 > 0.f) { s += s2 * expf(m2 - m); }
}

// v[r, j] = lm_j - logsumexp_i (a[r, i, j] + u[r, i])     (u null: 0).  grid (ceil((w+1)/32), rows), block (32, 8)
extern "C" __global__ void ot_col_lse(const __half* __restrict__ cv, const float* __restrict__ u, float* __restrict__ v,
                                      int w, int pos, float lm_a, float lm_d) {
    __shared__ float sm[8][32], ss[8][32];
    int r = blockIdx.y, j = blockIdx.x * 32 + threadIdx.x;
    const __half* base = cv + (long long)r * w * w;
    float m = __int_as_float(0xff800000), s = 0.f;
    if (j <= w) {
        for (int i = threadIdx.y; i <= w; i += 8) {
            float x = ot_a(base + (long long)i * w, i, j, w, pos);
            if (u) x += u[(long long)r * (w + 1) + i];
            lse_add(m, s, x);
        }
    }
    sm[threadIdx.y][threadIdx.x] = m;
    ss[threadIdx.y][threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.y == 0 && j <= w) {
        for (int k = 1; k < 8; ++k) lse_merge(m, s, sm[k][threadIdx.x], ss[k][threadIdx.x]);
        v[(long long)r * (w + 1) + j] = (j < w ? lm_a : lm_d) - (logf(s) + m);
    }
}

// u[r, i] = lm_i - logsumexp_j (a[r, i, j] + v[r, j]).  one warp per (r, i), i in [0, w]; block 256 (8 rows)
extern "C" __global__ void ot_row_lse(const __half* __restrict__ cv, const float* __restrict__ v, float* __restrict__ u,
                                      int rows, int w, int pos, float lm_a, float lm_d) {
    long long gw = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long long)rows * (w + 1)) return;
    int r = (int)(gw / (w + 1)), i = (int)(gw % (w + 1));
    const __half* row = cv + ((long long)r * w + (i < w ? i : 0)) * w;
    const float* vr = v + (long long)r * (w + 1);
    float m = __int_as_float(0xff800000), s = 0.f;
    for (int j = lane; j <= w; j += 32) lse_add(m, s, ot_a(row, i, j, w, pos) + vr[j]);
    for (int o = 16; o; o >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
        lse_merge(m, s, m2, s2);
    }
    if (lane == 0) u[(long long)r * (w + 1) + i] = (i < w ? lm_a : lm_d) - (logf(s) + m);
}

// transport plan p = exp(a + u + v + log_const) (masked -> 0) of pixel rows i < w, argmax over j, 5-tap soft-argmax:
// disp = i - (sum wt*idx + eps) / (sum wt + eps), conf = sum wt, occ = sum_j p.  one warp per (r, i)
__device__ __forceinline__ float ot_p(const __half* row, int i, int j, int w, int pos, float ui, const float* vr, float lc) {
    if (pos && j > i) return 0.f;
    return expf(((__half2float(row[j]) + ui) + vr[j]) + lc);
}

extern "C" __global__ void ot_final(const __half* __restrict__ cv, const float* __restrict__ u, const float* __restrict__ v,
                                    int rows, int w, int pos, float log_const,
                                    float* __restrict__ disp, float* __restrict__ conf, float* __restrict__ occ) {
    long long gw = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (long long)rows * w) return;
    int r = (int)(gw / w), i = (int)(gw % w);
    const __half* row = cv + ((long long)r * w + i) * w;
    const float* vr = v + (long long)r * (w + 1);
    float ui = u[(long long)r * (w + 1) + i];
    float best = -1.f, sum = 0.f;
    int bi = 0;
    for (int j = lane; j < w; j += 32) {
        float p = ot_p(row, i, j, w, pos, ui, vr, log_const);
        sum += p;
        if (p > best) { best = p; bi = j; }   // ascending j per lane: first max kept
    }
    for (int o = 16; o; o >>= 1) {
        float b2 = __shfl_xor_sync(0xffffffffu, best, o);
        int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (b2 > best || (b2 == best && i2 < bi)) { best = b2; bi = i2; }
    }
    if (lane == 0) {
        float c = 0.f, cl = 0.f;
        for (int k = -2; k <= 2; ++k) {
            int idx = bi + k;
            float wt = (idx >= 0 && idx < w) ? ot_p(row, i, idx, w, pos, ui, vr, log_const) : 0.f;
            c += wt;
            cl += wt * (float)idx;
        }
        const float eps = 1e-4f;
        long long o = (long long)r * w + i;
        disp[o] = (float)i - (cl + eps) / (c + eps);
        conf[o] = c;
        occ[o] = sum;
    }
}

// ------------------------------------------------------------------------------------------------ layer norm
// y[p, :] = (x[p, :] - mean) * rsqrt(var + eps), no affine (the attention blocks' LayerNorms), fp32 statistics
// (two passes over registers: mean, then centered variance). One warp per row, C % 4 == 0, C <= 1024.
extern "C" __global__ void ln_rows(const __half* __restrict__ x, __half* __restrict__ y, long long P, int C, float eps) {
    long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= P) return;
    const uint2* xr = reinterpret_cast<const uint2*>(x + row * C);
    int nv = C >> 2;
    float v[8][4];
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        int c = lane + 32 * k;
        if (c < nv) {
            uint2 raw = __ldg(xr + c);
            const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
            for (int e = 0; e < 4; ++e) { v[k][e] = __half2float(h[e]); s += v[k][e]; }
        }
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    float mean = s / C, q = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        if (lane + 32 * k < nv)
#pragma unroll
            for (int e = 0; e < 4; ++e) { float d = v[k][e] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    float rstd = rsqrtf(q / C + eps);
    uint2* yr = reinterpret_cast<uint2*>(y + row * C);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        int c = lane + 32 * k;
        if (c < nv) {
            __align__(8) __half o4[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) o4[e] = __float2half((v[k][e] - mean) * rstd);
            yr[c] = *reinterpret_cast<const uint2*>(o4);
        }
    }
}

// ------------------------------------------------------------------------------------------------ upsampling / fusion
// bilinear x2 (align_corners=False, = F.interpolate(scale_factor=2)) of NHWC x [N, H, W, C] into y with pixel stride ys
// (y may be a channel slot of a wider concat buffer). 8 channels per thread, fp32 math in torch's order.
// Output rows [oy0, oy0 + nrows) of each image only (row strips): y holds N x nrows x 2W pixels.
extern "C" __global__ void up2x_nhwc8(const __half* __restrict__ x, __half* __restrict__ y, int N, int H, int W, int C,
                                      long long ys, int oy0, int nrows) {
    const int CV = C / 8, Wo = 2 * W;
    GRID_STRIDE(i, (long long)N * nrows * Wo * CV) {
        int cv = (int)(i % CV);
        long long pix = i / CV;
        int ox = (int)(pix % Wo);
        long long t = pix / Wo;
        int oy = oy0 + (int)(t % nrows), n = (int)(t / nrows);
        float sy = fmaxf((oy + 0.5f) * 0.5f - 0.5f, 0.f), sx = fmaxf((ox + 0.5f) * 0.5f - 0.5f, 0.f);
        int y0 = (int)sy, x0 = (int)sx;
        int y1 = y0 + (y0 < H - 1), x1 = x0 + (x0 < W - 1);
        float ly = sy - y0, lx = sx - x0, hy = 1.f - ly, hx = 1.f - lx;
        const __half* b = x + (long long)n * H * W * C + cv * 8;
        uint4 r00 = *reinterpret_cast<const uint4*>(b + ((long long)y0 * W + x0) * C);
        uint4 r01 = *reinterpret_cast<const uint4*>(b + ((long long)y0 * W + x1) * C);
        uint4 r10 = *reinterpret_cast<const uint4*>(b + ((long long)y1 * W + x0) * C);
        uint4 r11 = *reinterpret_cast<const uint4*>(b + ((long long)y1 * W + x1) * C);
        const __half *a00 = reinterpret_cast<const __half*>(&r00), *a01 = reinterpret_cast<const __half*>(&r01);
        const __half *a10 = reinterpret_cast<const __half*>(&r10), *a11 = reinterpret_cast<const __half*>(&r11);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o[k] = __float2half(hy * (hx * __half2float(a00[k]) + lx * __half2float(a01[k])) +
                                ly * (hx * __half2float(a10[k]) + lx * __half2float(a11[k])));
        *reinterpret_cast<uint4*>(y + pix * ys + cv * 8) = *reinterpret_cast<const uint4*>(o);
    }
}

// FeatureFusion output with z0 / z1 read from a concat buffer (row stride zs): y = (y + b) + z1 + w * (z0 - z1)
extern "C" __global__ void gate_mix8s(__half* y, const float* __restrict__ b, const __half* __restrict__ w,
                                      const __half* __restrict__ z0, const __half* __restrict__ z1, long long P, int C,
                                      long long zs) {
    const int CV = C / 8;
    GRID_STRIDE(i, P * CV) {
        int c0 = (int)(i % CV) * 8;
        long long p = i / CV;
        uint4 ry = reinterpret_cast<uint4*>(y)[i];
        uint4 rw = reinterpret_cast<const uint4*>(w)[i];
        uint4 r0 = *reinterpret_cast<const uint4*>(z0 + p * zs + c0);
        uint4 r1 = *reinterpret_cast<const uint4*>(z1 + p * zs + c0);
        __half* hy = reinterpret_cast<__half*>(&ry);
        const __half* hw = reinterpret_cast<const __half*>(&rw);
        const __half* h0 = reinterpret_cast<const __half*>(&r0);
        const __half* h1 = reinterpret_cast<const __half*>(&r1);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float a = __half2float(h0[k]), c = __half2float(h1[k]);
            float v = __half2float(hy[k]) + (b ? b[c0 + k] : 0.f);
            hy[k] = __float2half(v + c + __half2float(hw[k]) * (a - c));
        }
        reinterpret_cast<uint4*>(y)[i] = ry;
    }
}

// AvgPool2d(2) (stride 2, floor) of NHWC x [N, H, W, C] -> y [N, H/2, W/2, C]; 8 channels per thread
extern "C" __global__ void pool2x_nhwc8(const __half* __restrict__ x, __half* __restrict__ y, int N, int H, int W, int C) {
    const int CV = C / 8, Ho = H / 2, Wo = W / 2;
    GRID_STRIDE(i, (long long)N * Ho * Wo * CV) {
        int cv = (int)(i % CV);
        long long pix = i / CV;
        int ox = (int)(pix % Wo);
        long long t = pix / Wo;
        int oy = (int)(t % Ho), n = (int)(t / Ho);
        const __half* b = x + (((long long)n * H + 2 * oy) * W + 2 * ox) * C + cv * 8;
        uint4 r00 = *reinterpret_cast<const uint4*>(b), r01 = *reinterpret_cast<const uint4*>(b + C);
        uint4 r10 = *reinterpret_cast<const uint4*>(b + (long long)W * C), r11 = *reinterpret_cast<const uint4*>(b + (long long)W * C + C);
        const __half *a00 = reinterpret_cast<const __half*>(&r00), *a01 = reinterpret_cast<const __half*>(&r01);
        const __half *a10 = reinterpret_cast<const __half*>(&r10), *a11 = reinterpret_cast<const __half*>(&r11);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o[k] = __float2half((((__half2float(a00[k]) + __half2float(a01[k])) + __half2float(a10[k])) + __half2float(a11[k])) / 4.f);
        reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<const uint4*>(o);
    }
}

// GroupNorm statistics of NHWC x [N, HW, C] (G groups of C/G channels, C/G % 8 == 0), DETERMINISTIC: thread = fixed
// 8-channel vector (blockDim = ppb * C/8). First pass sums values; with mean supplied, the second pass sums
// centered squared deviations. Fixed-order partials are reduced by the caller. grid (blocks, N), blockDim <= 256.
extern "C" __global__ void gn_stats(const __half* __restrict__ x, double* __restrict__ part, const float* __restrict__ mean, long long HW, int C, int G) {
    __shared__ float ss[2][256];
    int n = blockIdx.y;
    const int CV = C / 8, cpg = C / G, ppb = blockDim.x / CV;
    int cv = threadIdx.x % CV, pl = threadIdx.x / CV;
    float s = 0.f, q = 0.f;
    if (pl < ppb) {
        const __half* xb = x + (long long)n * HW * C + cv * 8;
        for (long long p = blockIdx.x * (long long)ppb + pl; p < HW; p += (long long)gridDim.x * ppb) {
            uint4 raw = *reinterpret_cast<const uint4*>(xb + p * C);
            const __half* h = reinterpret_cast<const __half*>(&raw);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v = __half2float(h[k]);
                if (mean) { float d = v - mean[n * G + (cv * 8 + k) / cpg]; q += d * d; }
                else s += v;
            }
        }
    }
    ss[0][threadIdx.x] = s;
    ss[1][threadIdx.x] = q;
    __syncthreads();
    if (threadIdx.x < G) {
        int g = threadIdx.x, vpg = cpg / 8;          // channel vectors per group: cv in [g*vpg, (g+1)*vpg)
        double S = 0.0, Q = 0.0;
        for (int p = 0; p < ppb; ++p)
            for (int c = g * vpg; c < (g + 1) * vpg; ++c) { S += ss[0][p * CV + c]; Q += ss[1][p * CV + c]; }
        long long o = (((long long)n * gridDim.x + blockIdx.x) * G + g) * 2;
        part[o] = S;
        part[o + 1] = Q;
    }
}

// Center before scaling to avoid cancellation for nearly constant high-offset groups.
extern "C" __global__ void affine_nc8(__half* x, const float* __restrict__ s, const float* __restrict__ t, const float* __restrict__ mean, long long HW, int C,
                                      int N, int G) {
    const int CV = C / 8;
    GRID_STRIDE(i, (long long)N * HW * CV) {
        int c0 = (int)(i % CV) * 8;
        int n = (int)(i / (HW * CV));
        uint4 raw = reinterpret_cast<uint4*>(x)[i];
        __half* h = reinterpret_cast<__half*>(&raw);
#pragma unroll
        for (int k = 0; k < 8; ++k) h[k] = __float2half((__half2float(h[k]) - mean[n * G + (c0 + k) / (C / G)]) * s[n * C + c0 + k] + t[c0 + k]);
        reinterpret_cast<uint4*>(x)[i] = raw;
    }
}

// convex upsampling (S2M2 UpsampleMask4x / 1x): out[c, Y, X] = sum_k softmax_k(mask[k, Y, X]) * Xp[c, Y/f + ky, X/f + kx]
// with Xp = replicate-padded X (fp32 [c, h, w]), mask fp16 NHWC [H, W, 9] (H = f*h), out fp32 [c, H, W]; f = 4 or 1
extern "C" __global__ void convex_up(const float* __restrict__ X, const __half* __restrict__ mask, float* __restrict__ out,
                                     int c, int h, int w, int f) {
    const int H = h * f, W = w * f;
    GRID_STRIDE(i, (long long)H * W) {
        int Xo = (int)(i % W), Yo = (int)(i / W);
        int y = Yo / f, x = Xo / f;
        const __half* m = mask + i * 9;
        float mv[9], mx = -1e30f;
#pragma unroll
        for (int k = 0; k < 9; ++k) { mv[k] = __half2float(m[k]); mx = fmaxf(mx, mv[k]); }
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < 9; ++k) { mv[k] = expf(mv[k] - mx); sum += mv[k]; }
        for (int cc = 0; cc < c; ++cc) {
            const float* Xc = X + (long long)cc * h * w;
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                int yy = min(max(y + k / 3 - 1, 0), h - 1), xx = min(max(x + k % 3 - 1, 0), w - 1);
                acc += (mv[k] / sum) * Xc[(long long)yy * w + xx];
            }
            out[(long long)cc * H * W + i] = acc;
        }
    }
}
