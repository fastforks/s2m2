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
// with Xp = replicate-padded X (fp32 [c, h, w]), mask fp16 NHWC [H, W, 9] (H = f*h), out fp32 [c, H, W]; f = 4 or 1.
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

// ------------------------------------------------------------------------------------------------ on-the-fly cost volume
// Refinement lookup without the [h, w, w] correlation volume: cv[y, i, j] = fp16(dot(f0[y, i, :], f1[y, j, :])) with
// f = fp16(LayerNorm(x)) recomputed from the raw transformer output x [2, h, w, C] fp16 (per-pixel fp32 mean / rstd
// in ms [2, h*w, 2], affine g, b fp32 [C]): only the ~20 right-image columns a pixel's 2 x 9 samples touch, fp32 dots
// rounded to fp16 like the materialized cuBLAS volume. Same sample positions / interpolation / zero padding as
// FastCostVolume (inference.py): level 1: x1 = (x*s1 - .5 - d*s1) + k*s1, level 2 (avg_pool1d(2) of cv, read as pair
// means): x2 = (x*hs2 - .5 - d*hs2) + k*s2 (hs2 = .5*s2), k = -4..4; v = (v1 - v0) * t + v0. disp fp32 [h*w];
// out1 / out2 fp16 [h*w, 9]. One warp per pixel (lane = channel pairs lane*2 + 64k); column dots cached in smem.
#define OTF_MAXCOL 64
template <int NC>
__device__ __forceinline__ void otf_ln(const __half* __restrict__ xp, float mu, float rs, const float2* g, const float2* b,
                                       int lane, float2* a) {
    const __half2* x2 = reinterpret_cast<const __half2*>(xp) + lane;
#pragma unroll
    for (int k = 0; k < NC; ++k) {
        float2 v = __half22float2(x2[k * 32]);
        a[k].x = __half2float(__float2half_rn(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(v.x, mu), rs), g[k].x), b[k].x)));
        a[k].y = __half2float(__float2half_rn(__fadd_rn(__fmul_rn(__fmul_rn(__fsub_rn(v.y, mu), rs), g[k].y), b[k].y)));
    }
}

template <int NC>
__device__ __forceinline__ float otf_dot(const __half* __restrict__ xr, float2 msr, const float2* g, const float2* b,
                                         int lane, const float2* a) {
    float2 f[NC];
    otf_ln<NC>(xr, msr.x, msr.y, g, b, lane, f);
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < NC; ++k) {
        s = fmaf(a[k].x, f[k].x, s);
        s = fmaf(a[k].y, f[k].y, s);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return __half2float(__float2half_rn(s));
}

template <int NC>
__device__ __forceinline__ void otf_body(const __half* __restrict__ x, const float2* __restrict__ ms,
                                         const float* __restrict__ gw, const float* __restrict__ gb,
                                         const float* __restrict__ disp, __half* __restrict__ out1, __half* __restrict__ out2,
                                         int h, int w, int C, float s1, float hs2, float s2, float (*cache)[OTF_MAXCOL]) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int W2 = w >> 1;
    const long long HW = (long long)h * w;
    float2 g[NC], b[NC];
#pragma unroll
    for (int k = 0; k < NC; ++k) {
        g[k] = make_float2(gw[lane * 2 + 64 * k], gw[lane * 2 + 64 * k + 1]);
        b[k] = make_float2(gb[lane * 2 + 64 * k], gb[lane * 2 + 64 * k + 1]);
    }
    const __half* x1 = x + HW * C;
    const float2* ms1 = ms + HW;
    long long nwarps = (long long)gridDim.x * 8;
    for (long long p = blockIdx.x * 8LL + warp; p < HW; p += nwarps) {
        const int y = (int)(p / w), xq = (int)(p % w);
        const float d = disp[p];
        float2 a[NC];
        { float2 m = ms[p]; otf_ln<NC>(x + p * C, m.x, m.y, g, b, lane, a); }
        const float xf = (float)xq;
        const float b1 = __fsub_rn(__fmul_rn(xf, s1), 0.5f), c1 = __fsub_rn(b1, __fmul_rn(d, s1));
        const float b2 = __fsub_rn(__fmul_rn(xf, hs2), 0.5f), c2 = __fsub_rn(b2, __fmul_rn(d, hs2));
        // column window touched by both levels (positions are monotone in k)
        const int l1 = (int)floorf(__fadd_rn(c1, __fmul_rn(-4.f, s1))), h1 = (int)floorf(__fadd_rn(c1, __fmul_rn(4.f, s1))) + 1;
        const int l2 = 2 * (int)floorf(__fadd_rn(c2, __fmul_rn(-4.f, s2))), h2 = 2 * ((int)floorf(__fadd_rn(c2, __fmul_rn(4.f, s2))) + 1) + 1;
        const int jlo = max(0, min(l1, l2)), jhi = min(w - 1, max(h1, h2));
        const int n = jhi - jlo + 1;
        const long long rowp = (long long)y * w;
        const bool cached = n <= OTF_MAXCOL;
        if (cached) {
            for (int j = 0; j < n; ++j) {
                float v = otf_dot<NC>(x1 + (rowp + jlo + j) * C, ms1[rowp + jlo + j], g, b, lane, a);
                if (lane == 0) cache[warp][j] = v;
            }
        }
        __syncwarp();
        // lanes 0..8: level 1 sample k = lane - 4; lanes 9..17: level 2. Uncached (n > 64, never for w >= 8): direct dots.
        const bool lv1 = lane < 9;
        const int kk = (lv1 ? lane : lane - 9) - 4;
        const float xs = lv1 ? __fadd_rn(c1, __fmul_rn((float)kk, s1)) : __fadd_rn(c2, __fmul_rn((float)kk, s2));
        const float x0 = floorf(xs), t = __fsub_rn(xs, x0);
        const int i0 = (int)x0;
        float v[2] = {0.f, 0.f};
        if (cached && lane < 18) {
            for (int q = 0; q < 2; ++q) {
                int i = i0 + q;
                if (lv1) { if (i >= 0 && i <= w - 1) v[q] = cache[warp][i - jlo]; }
                else if (i >= 0 && i <= W2 - 1)
                    v[q] = __fmul_rn(__fadd_rn(cache[warp][2 * i - jlo], cache[warp][2 * i + 1 - jlo]), 0.5f);
            }
        }
        if (!cached) {   // slow path: every sample lane's columns, warp-wide dots
            for (int src = 0; src < 18; ++src) {
                int ii0 = __shfl_sync(0xffffffffu, i0, src);
                bool l = src < 9;
                for (int q = 0; q < 2; ++q) {
                    int i = ii0 + q;
                    float val = 0.f;
                    if (l && i >= 0 && i <= w - 1)
                        val = otf_dot<NC>(x1 + (rowp + i) * C, ms1[rowp + i], g, b, lane, a);
                    else if (!l && i >= 0 && i <= W2 - 1)
                        val = __fmul_rn(__fadd_rn(otf_dot<NC>(x1 + (rowp + 2 * i) * C, ms1[rowp + 2 * i], g, b, lane, a),
                                                  otf_dot<NC>(x1 + (rowp + 2 * i + 1) * C, ms1[rowp + 2 * i + 1], g, b, lane, a)), 0.5f);
                    if (lane == src) v[q] = val;
                }
            }
        }
        const float res = __fadd_rn(__fmul_rn(__fsub_rn(v[1], v[0]), t), v[0]);
        if (lane < 9) out1[p * 9 + lane] = __float2half_rn(res);
        else if (lane < 18) out2[p * 9 + lane - 9] = __float2half_rn(res);
        __syncwarp();
    }
}

extern "C" __global__ void cv_lookup_otf(const __half* __restrict__ x, const float2* __restrict__ ms,
                                         const float* __restrict__ gw, const float* __restrict__ gb,
                                         const float* __restrict__ disp, __half* __restrict__ out1, __half* __restrict__ out2,
                                         int h, int w, int C, float s1, float hs2, float s2) {
    __shared__ float cache[8][OTF_MAXCOL];
    switch (C >> 6) {   // half2 per lane
        case 1: otf_body<1>(x, ms, gw, gb, disp, out1, out2, h, w, C, s1, hs2, s2, cache); break;
        case 2: otf_body<2>(x, ms, gw, gb, disp, out1, out2, h, w, C, s1, hs2, s2, cache); break;
        case 3: otf_body<3>(x, ms, gw, gb, disp, out1, out2, h, w, C, s1, hs2, s2, cache); break;
        case 4: otf_body<4>(x, ms, gw, gb, disp, out1, out2, h, w, C, s1, hs2, s2, cache); break;
        case 6: otf_body<6>(x, ms, gw, gb, disp, out1, out2, h, w, C, s1, hs2, s2, cache); break;
        default: otf_body<8>(x, ms, gw, gb, disp, out1, out2, h, w, C, s1, hs2, s2, cache); break;
    }
}

// ------------------------------------------------------------------------------------------------ attention, head_dim 384
// Flash-style fused attention for S2M2-XL's 1-head level-0 attention (head_dim 384 > flash/SDPA's 256): o = softmax(q k^T
// * scale) v for B = n*H independent sequences of length L. fp16 inputs, fp16-accumulate tensor-core MMAs for q k^T
// and p v (the precision class of the explicit fp16-accumulate cuBLAS path it replaces), online softmax statistics in
// fp32; p is pre-scaled by 2^-5 so the fp16 accumulator of sum_j p_j v_j has 32x headroom (unnormalized p <= 1).
// Block = 4 warps x 16 query rows; key tiles of 32 rows (K and V tiles in shared memory, cp.async, single buffered
// with the V load overlapping q k^T and the next K load overlapping p v). q fragments live in registers.
// Strides (elements): sequence b = (b / H) * sn + (b % H) * sh, row l * sl (q, k, v share them; o has its own).
#define AT_D 384
#define AT_BN 32
#define AT_LD (AT_D + 8)
__device__ __forceinline__ void cp16(void* s, const void* g, bool ok) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(g), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x4t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma16(unsigned* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

extern "C" __global__ void __launch_bounds__(128) attn_d384(
        const __half* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v, __half* __restrict__ o,
        long long sn, long long sh, long long sl, long long osn, long long osh, long long osl, int H, int L, float scale) {
    extern __shared__ __align__(16) __half smem[];
    __half* sK = smem;                    // [AT_BN][AT_LD]
    __half* sV = smem + AT_BN * AT_LD;    // [AT_BN][AT_LD]   (sK | sV = [64][AT_LD] q staging at start)
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int b = blockIdx.y, m0 = blockIdx.x * 64;
    const long long base = (long long)(b / H) * sn + (long long)(b % H) * sh;
    const long long obase = (long long)(b / H) * osn + (long long)(b % H) * osh;
    const __half* qb = q + base; const __half* kb = k + base; const __half* vb = v + base;
    // q tile -> smem -> registers
    for (int c = tid; c < 64 * (AT_D / 8); c += 128) {
        int r = c / (AT_D / 8), cc = (c % (AT_D / 8)) * 8;
        bool ok = m0 + r < L;
        cp16(smem + r * AT_LD + cc, qb + (long long)(ok ? m0 + r : 0) * sl + cc, ok);
    }
    cp_commit(); cp_wait<0>(); __syncthreads();
    unsigned qf[AT_D / 16][4];
#pragma unroll
    for (int kk = 0; kk < AT_D / 16; ++kk)
        ldm_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], smem + (warp * 16 + (lane & 15)) * AT_LD + kk * 16 + (lane >> 4) * 8);
    __syncthreads();
    auto load_tile = [&](__half* s, const __half* g, int n0) {
        for (int c = tid; c < AT_BN * (AT_D / 8); c += 128) {
            int r = c / (AT_D / 8), cc = (c % (AT_D / 8)) * 8;
            bool ok = n0 + r < L;
            cp16(s + r * AT_LD + cc, g + (long long)(ok ? n0 + r : 0) * sl + cc, ok);
        }
        cp_commit();
    };
    unsigned oacc[AT_D / 8][2];
#pragma unroll
    for (int i = 0; i < AT_D / 8; ++i) { oacc[i][0] = 0u; oacc[i][1] = 0u; }
    float mA = -INFINITY, mB = -INFINITY, lA = 0.f, lB = 0.f;
    const float sc = scale * 1.4426950408889634f;   // exp2 domain
    const int T = (L + AT_BN - 1) / AT_BN;
    load_tile(sK, kb, 0);
    for (int t = 0; t < T; ++t) {
        const int n0 = t * AT_BN;
        load_tile(sV, vb, n0);
        cp_wait<1>(); __syncthreads();             // K tile t ready
        unsigned s[AT_BN / 8][2];
#pragma unroll
        for (int i = 0; i < AT_BN / 8; ++i) { s[i][0] = 0u; s[i][1] = 0u; }
#pragma unroll
        for (int kk = 0; kk < AT_D / 16; ++kk) {
#pragma unroll
            for (int np = 0; np < AT_BN / 16; ++np) {
                unsigned b0, b1, b2, b3;
                const int mid = lane >> 3;
                ldm_x4(b0, b1, b2, b3, sK + (np * 16 + (mid >> 1) * 8 + (lane & 7)) * AT_LD + kk * 16 + (mid & 1) * 8);
                mma16(s[2 * np], qf[kk], b0, b1);
                mma16(s[2 * np + 1], qf[kk], b2, b3);
            }
        }
        __syncthreads();                           // all warps done with sK
        if (t + 1 < T) load_tile(sK, kb, n0 + AT_BN);
        // online softmax (rows A = lane/4, B = lane/4 + 8 of the warp's 16)
        float pa[AT_BN / 8][2], pb[AT_BN / 8][2];
        float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
        for (int i = 0; i < AT_BN / 8; ++i) {
            float2 xa = __half22float2(u2h(s[i][0])), xb = __half22float2(u2h(s[i][1]));
            int c = n0 + i * 8 + (lane & 3) * 2;
            pa[i][0] = c < L ? xa.x * sc : -INFINITY; pa[i][1] = c + 1 < L ? xa.y * sc : -INFINITY;
            pb[i][0] = c < L ? xb.x * sc : -INFINITY; pb[i][1] = c + 1 < L ? xb.y * sc : -INFINITY;
            mxA = fmaxf(mxA, fmaxf(pa[i][0], pa[i][1])); mxB = fmaxf(mxB, fmaxf(pb[i][0], pb[i][1]));
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffffu, mxA, off));
            mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffffu, mxB, off));
        }
        const float nA = fmaxf(mA, mxA), nB = fmaxf(mB, mxB);
        const float aA = exp2f(mA - nA), aB = exp2f(mB - nB);   // mA = -inf at t = 0 -> 0
        mA = nA; mB = nB;
        float rsA = 0.f, rsB = 0.f;
        unsigned pf[AT_BN / 8][2];
#pragma unroll
        for (int i = 0; i < AT_BN / 8; ++i) {
            float p0 = exp2f(pa[i][0] - nA - 5.f), p1 = exp2f(pa[i][1] - nA - 5.f);
            float p2 = exp2f(pb[i][0] - nB - 5.f), p3 = exp2f(pb[i][1] - nB - 5.f);
            __half2 ha = __floats2half2_rn(p0, p1), hb = __floats2half2_rn(p2, p3);
            float2 fa = __half22float2(ha), fb = __half22float2(hb);   // row sums of the rounded p (consistent with p v)
            rsA += fa.x + fa.y; rsB += fb.x + fb.y;
            pf[i][0] = h2u(ha); pf[i][1] = h2u(hb);
        }
        rsA += __shfl_xor_sync(0xffffffffu, rsA, 1); rsA += __shfl_xor_sync(0xffffffffu, rsA, 2);
        rsB += __shfl_xor_sync(0xffffffffu, rsB, 1); rsB += __shfl_xor_sync(0xffffffffu, rsB, 2);
        lA = lA * aA + rsA; lB = lB * aB + rsB;
        if (t > 0) {
            const __half2 hA = __float2half2_rn(aA), hB = __float2half2_rn(aB);
#pragma unroll
            for (int i = 0; i < AT_D / 8; ++i) {
                oacc[i][0] = h2u(__hmul2(u2h(oacc[i][0]), hA));
                oacc[i][1] = h2u(__hmul2(u2h(oacc[i][1]), hB));
            }
        }
        if (t + 1 < T) cp_wait<1>(); else cp_wait<0>();
        __syncthreads();                           // V tile t ready
#pragma unroll
        for (int j = 0; j < AT_BN / 16; ++j) {
            unsigned a[4] = {pf[2 * j][0], pf[2 * j][1], pf[2 * j + 1][0], pf[2 * j + 1][1]};
#pragma unroll
            for (int np = 0; np < AT_D / 16; ++np) {
                unsigned b0, b1, b2, b3;
                const int mid = lane >> 3;
                ldm_x4t(b0, b1, b2, b3, sV + (j * 16 + (mid & 1) * 8 + (lane & 7)) * AT_LD + np * 16 + (mid >> 1) * 8);
                mma16(oacc[2 * np], a, b0, b1);
                mma16(oacc[2 * np + 1], a, b2, b3);
            }
        }
        __syncthreads();                           // all warps done with sV
    }
    const int rA = m0 + warp * 16 + (lane >> 2), rB = rA + 8;
    const float iA = 1.f / lA, iB = 1.f / lB;
#pragma unroll
    for (int i = 0; i < AT_D / 8; ++i) {
        const int c = i * 8 + (lane & 3) * 2;
        float2 xa = __half22float2(u2h(oacc[i][0])), xb = __half22float2(u2h(oacc[i][1]));
        if (rA < L) *reinterpret_cast<__half2*>(o + obase + (long long)rA * osl + c) = __floats2half2_rn(xa.x * iA, xa.y * iA);
        if (rB < L) *reinterpret_cast<__half2*>(o + obase + (long long)rB * osl + c) = __floats2half2_rn(xb.x * iB, xb.y * iB);
    }
}
