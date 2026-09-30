/***************************************************************************
 *                     NNSD CUDA GPU Training Backend                       *
 *                                                                           *
 * Implements batched FeedForward + BackProp entirely on the GPU using       *
 * custom tiled SGEMM kernels (no cuBLAS dependency).                        *
 *                                                                           *
 * Weight layout (device):  w_d[l]  →  (neurons[l]+1) × neurons[l+1]       *
 *                          flat row-major ("from" index is the row).       *
 *                                                                           *
 * Activation buffers:      a_d[l]  →  batch × (neurons[l]+1) for l < L-1  *
 *                                      batch × neurons[l]   for l = L-1    *
 * Pre-activation buffers:  z_d[l]  →  batch × neurons[l]   for l ≥ 1      *
 * Delta buffers:           delta_d[l] → batch × neurons[l] for l ≥ 1      *
 ***************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "common.h"
#include "log_utils.h"
#include "nnsd.h"
#include "nn_cuda.h"
}

/* ── tuning constants ─────────────────────────────────────────────────── */

#define TILE        16
#define CUDA_BATCH  512

/* ── CudaContext (opaque to callers) ─────────────────────────────────── */

typedef struct {
    /* Device weight matrices: w_d[l] → (neurons[l]+1) × neurons[l+1] */
    float *w_d[MAX_LAYERS];
    float *m_d[MAX_LAYERS];     /* Adam 1st moment, same shape */
    float *v_d[MAX_LAYERS];     /* Adam 2nd moment, same shape */
    float *prev_d[MAX_LAYERS];  /* SGD momentum,    same shape */
    float *dw_d[MAX_LAYERS];    /* gradient accumulator */

    /* Activation / delta buffers (batch-sized) */
    float *z_d[MAX_LAYERS];     /* pre-activation:  batch × neurons[l] */
    float *a_d[MAX_LAYERS];     /* post-activation: batch × a_cols[l]  */
    float *delta_d[MAX_LAYERS]; /* deltas:          batch × neurons[l] */

    /* Full dataset uploaded once */
    float *inputs_d;   /* N × (neurons[0]+1), pre-scaled with bias = 1 */
    float *labels_d;   /* N × neurons[L-1], one-hot (or scaled)        */

    /* Per-epoch stats (device scalars) */
    int   *correct_d;
    float *sse_d;

    int  batch_size;
    int  n_samples;
    int  n_layers;
    int  neurons[MAX_LAYERS];
    int  fn;          /* activation function selector */
    long adamT;
    unsigned long long epoch_count; /* incremented each CudaTrainEpoch call */
} CudaContext;

/* ── device activation / derivative helpers ──────────────────────────── */

__device__ static float nn_activate_d(float z, int fn) {
    switch (fn) {
    case 1: return tanhf(z);
    case 2: return z > 0.0f ? z : 0.0f;
    case 3: return z > 0.0f ? z : 0.01f * z;
    default:
        if (z < -8.0f) z = -8.0f; else if (z > 8.0f) z = 8.0f;
        return 1.0f / (1.0f + expf(-z));
    }
}

__device__ static float nn_deriv_d(float a, int fn) {
    switch (fn) {
    case 1: return 1.0f - a * a;
    case 2: return a > 0.0f ? 1.0f : 0.0f;
    case 3: return a >= 0.0f ? 1.0f : 0.01f;
    default: return a * (1.0f - a);
    }
}


/* ── tiled SGEMM kernels (row-major) ─────────────────────────────────── */

/*
 * gemm_nn: C[M×N] = alpha * A[M×K] @ B[K×N] + beta * C[M×N]
 */
__global__ static void gemm_nn(const float *A, int lda,
                                const float *B, int ldb,
                                float       *C, int ldc,
                                int M, int N, int K,
                                float alpha, float beta)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * lda + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * ldb + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * ldc + col] = alpha * acc + (beta != 0.0f ? beta * C[row * ldc + col] : 0.0f);
}

/*
 * gemm_nt: C[M×N] = alpha * A[M×K] @ B[N×K]^T + beta * C[M×N]
 * Used for backprop delta propagation: delta[l] = delta[l+1] @ W[l]^T
 */
__global__ static void gemm_nt(const float *A, int lda,
                                const float *B, int ldb,
                                float       *C, int ldc,
                                int M, int N, int K,
                                float alpha, float beta)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bCol = t * TILE + threadIdx.y; /* B is [N×K]; B^T[k][col] = B[col][k] */
        int bRow = blockIdx.x * TILE + threadIdx.x; /* column of C → row of B */
        As[threadIdx.y][threadIdx.x] = (row  < M && aCol < K) ? A[row  * lda + aCol]  : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && bCol < K) ? B[bRow * ldb + bCol]  : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * ldc + col] = alpha * acc + (beta != 0.0f ? beta * C[row * ldc + col] : 0.0f);
}

/*
 * gemm_tn: C[M×N] = alpha * A[K×M]^T @ B[K×N] + beta * C[M×N]
 * Used for gradient accumulation: dW[l] += scale * a[l]^T @ delta[l+1]
 */
__global__ static void gemm_tn(const float *A, int lda,
                                const float *B, int ldb,
                                float       *C, int ldc,
                                int M, int N, int K,
                                float alpha, float beta)
{
    __shared__ float As[TILE][TILE]; /* As[ty][tx] = A[k_off+ty][row_block+tx] */
    __shared__ float Bs[TILE][TILE]; /* Bs[ty][tx] = B[k_off+ty][col]          */
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int k_off     = t * TILE;
        int rowBlock  = blockIdx.y * TILE;
        As[threadIdx.y][threadIdx.x] =
            (k_off + threadIdx.y < K && rowBlock + threadIdx.x < M)
            ? A[(k_off + threadIdx.y) * lda + rowBlock + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (k_off + threadIdx.y < K && col < N)
            ? B[(k_off + threadIdx.y) * ldb + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; k++) acc += As[k][threadIdx.y] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * ldc + col] = alpha * acc + (beta != 0.0f ? beta * C[row * ldc + col] : 0.0f);
}

/* ── element-wise kernels ─────────────────────────────────────────────── */

/*
 * activate_bias_k: read z[batch×n_l], write activated values to a[batch×a_stride],
 * and optionally set the bias column a[row][n_l] = 1.0.
 */
__global__ static void activate_bias_k(const float *z, int z_stride,
                                        float       *a, int a_stride,
                                        int batch, int n_l, int add_bias, int fn)
{
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * (n_l + (add_bias ? 1 : 0));
    if (tid >= total) return;
    int row  = tid / (n_l + (add_bias ? 1 : 0));
    int col  = tid % (n_l + (add_bias ? 1 : 0));
    if (col < n_l)
        a[row * a_stride + col] = nn_activate_d(z[row * z_stride + col], fn);
    else if (add_bias)
        a[row * a_stride + col] = 1.0f;
}

/*
 * output_delta_k: flat kernel over batch*n_out elements.
 * delta[i] = (label[i] - pred[i]) * deriv(pred[i])
 */
__global__ static void output_delta_k(const float *pred, const float *label,
                                       float *delta, int total, int fn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    float p = pred[tid];
    delta[tid] = (label[tid] - p) * nn_deriv_d(p, fn);
}

/*
 * apply_deriv_k: multiply delta in-place by derivative of activation.
 * a and delta may have different column strides.
 */
__global__ static void apply_deriv_k(const float *a, int a_stride,
                                      float *delta,   int d_stride,
                                      int batch, int n_l, int fn)
{
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_l;
    if (tid >= total) return;
    int row = tid / n_l;
    int col = tid % n_l;
    delta[row * d_stride + col] *= nn_deriv_d(a[row * a_stride + col], fn);
}

/*
 * apply_deriv_dropout_k: backprop derivative step when inverted dropout was
 * applied in the forward pass.
 *
 * Dropped neurons (a == 0 exactly) have their delta zeroed.
 * Surviving neurons had their activation scaled by 1/keep_prob; undo that
 * scaling before computing the derivative so sigmoid's a*(1-a) is evaluated
 * on a value in [0,1] rather than [0, 1/keep_prob].
 */
__global__ static void apply_deriv_dropout_k(const float *a, int a_stride,
                                               float *delta,   int d_stride,
                                               int batch, int n_l, int fn,
                                               float keep_prob)
{
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_l;
    if (tid >= total) return;
    int   row = tid / n_l;
    int   col = tid % n_l;
    float av  = a[row * a_stride + col];
    if (av == 0.0f) {
        delta[row * d_stride + col] = 0.0f;
    } else {
        float a_orig = av * keep_prob;   /* undo inverted-dropout scaling */
        delta[row * d_stride + col] *= nn_deriv_d(a_orig, fn);
    }
}

/*
 * adam_update_k: Adam weight update.  Resets dw to 0 after update.
 */
__global__ static void adam_update_k(float *w, float *m, float *v, float *dw,
                                      float lr, float b1, float b2, float eps,
                                      float bc1, float bc2, float decay, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float g  = dw[tid];
    float mi = b1 * m[tid] + (1.0f - b1) * g;
    float vi = b2 * v[tid] + (1.0f - b2) * g * g;
    m[tid]   = mi;
    v[tid]   = vi;
    w[tid]   = w[tid] * (1.0f - decay)
             + lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
    dw[tid]  = 0.0f;
}

/*
 * sgd_update_k: SGD + momentum weight update.  Resets dw to 0 after update.
 */
__global__ static void sgd_update_k(float *w, float *prev, float *dw,
                                     float lr, float mom, float decay, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float delta  = lr * dw[tid] + mom * prev[tid];
    prev[tid]    = delta;
    w[tid]       = w[tid] * (1.0f - decay) + delta;
    dw[tid]      = 0.0f;
}

/*
 * stats_k: per-sample accuracy (argmax) and SSE accumulation.
 */
__global__ static void stats_k(const float *pred, const float *label,
                                int batch, int n_out,
                                int *correct, float *sse)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) return;
    const float *p = pred  + sample * n_out;
    const float *l = label + sample * n_out;
    float s = 0.0f;
    for (int j = 0; j < n_out; j++) { float d = l[j] - p[j]; s += d * d; }
    atomicAdd(sse, s);
    /* For single-output networks use a threshold comparison (pred >= 0.5)
     * rather than argmax, which is degenerate and always returns "correct". */
    int pc, tc;
    if (n_out == 1) {
        pc = (p[0] >= 0.5f) ? 1 : 0;
        tc = (l[0] >= 0.5f) ? 1 : 0;
    } else {
        pc = 0; tc = 0;
        for (int j = 1; j < n_out; j++) {
            if (p[j] > p[pc]) pc = j;
            if (l[j] > l[tc]) tc = j;
        }
    }
    if (pc == tc) atomicAdd(correct, 1);
}

/*
 * dropout_k: inverted dropout on hidden-layer activations.
 * Zeros neurons with probability (1-keep_prob) and scales survivors by
 * 1/keep_prob so expected values are preserved at test time.
 * Uses a deterministic xorshift64* hash of (epoch_seed XOR tid) — no
 * curand state needed.
 */
__global__ static void dropout_k(float *a, int stride, int batch, int n_l,
                                  float keep_prob, unsigned long long epoch_seed)
{
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_l;
    if (tid >= total) return;
    int row = tid / n_l;
    int col = tid % n_l;
    unsigned long long x = (epoch_seed ^ (unsigned long long)tid) * 6364136223846793005ULL + 1442695040888963407ULL;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    float rnd = (float)(x & 0xFFFFFFFFULL) * 2.3283064365386963e-10f;
    if (rnd >= keep_prob)
        a[row * stride + col] = 0.0f;
    else
        a[row * stride + col] /= keep_prob;
}


/* ── kernel launch wrappers (host side) ──────────────────────────────── */

static void launch_gemm_nn(const float *A, int lda,
                            const float *B, int ldb,
                            float       *C, int ldc,
                            int M, int N, int K, float alpha, float beta)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_nn<<<grid, block>>>(A, lda, B, ldb, C, ldc, M, N, K, alpha, beta);
}

static void launch_gemm_nt(const float *A, int lda,
                            const float *B, int ldb,
                            float       *C, int ldc,
                            int M, int N, int K, float alpha, float beta)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_nt<<<grid, block>>>(A, lda, B, ldb, C, ldc, M, N, K, alpha, beta);
}

static void launch_gemm_tn(const float *A, int lda,
                            const float *B, int ldb,
                            float       *C, int ldc,
                            int M, int N, int K, float alpha, float beta)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tn<<<grid, block>>>(A, lda, B, ldb, C, ldc, M, N, K, alpha, beta);
}

/* ── weight pack / unpack helpers ────────────────────────────────────── */

/* Flatten jagged host weights[l][k][j] → contiguous row-major array. */
static float *flatten_weights(NetStruct *net, int l)
{
    int n_in1 = net->neurons[l] + 1;
    int n_out = net->neurons[l + 1];
    float *flat = (float *)malloc(n_in1 * n_out * sizeof(float));
    for (int k = 0; k < n_in1; k++)
        memcpy(flat + k * n_out, net->weights[l][k], n_out * sizeof(float));
    return flat;
}

/* Copy contiguous flat array back to jagged host weights[l][k][j]. */
static void unflatten_weights(NetStruct *net, int l, const float *flat)
{
    int n_in1 = net->neurons[l] + 1;
    int n_out = net->neurons[l + 1];
    for (int k = 0; k < n_in1; k++)
        memcpy(net->weights[l][k], flat + k * n_out, n_out * sizeof(float));
}

/* ── CudaInit ─────────────────────────────────────────────────────────── */

extern "C" void CudaFree(void *opaque, int n_layers);   /* forward decl */

extern "C"
void *CudaInit(ThreadStruct *netPtr)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        LOG_INFO("CUDA: no GPU found, falling back to CPU path");
        return NULL;
    }

    NetStruct  *net = netPtr->netStructPtr;
    FileStruct *fs  = netPtr->fileStructPtr;
    int L   = net->layers;
    int N   = fs->numInputRows;
    int n0  = net->neurons[0];
    int n0p = n0 + 1;                              /* input width with bias */
    int nL  = net->neurons[L - 1];                 /* output width           */

    CudaContext *ctx = (CudaContext *)calloc(1, sizeof(CudaContext));
    if (!ctx) return NULL;

    ctx->batch_size = CUDA_BATCH;
    ctx->n_samples  = N;
    ctx->n_layers   = L;
    ctx->fn         = net->activationFunction;
    ctx->adamT      = net->adamT;
    for (int l = 0; l < L; l++) ctx->neurons[l] = net->neurons[l];

    /* ── allocate and upload weight / moment / grad buffers ── */
    for (int l = 0; l < L - 1; l++) {
        int n_in1 = net->neurons[l] + 1;
        int n_out = net->neurons[l + 1];
        size_t wsz = (size_t)n_in1 * n_out * sizeof(float);

        float *h_w = flatten_weights(net, l);
        cudaMalloc(&ctx->w_d[l], wsz);
        cudaMemcpy(ctx->w_d[l], h_w, wsz, cudaMemcpyHostToDevice);
        free(h_w);

        /* Adam moments (copy from host if already trained, else zero) */
        cudaMalloc(&ctx->m_d[l], wsz);
        cudaMalloc(&ctx->v_d[l], wsz);
        if (net->useAdam && net->adamM[l] && net->adamM[l][0]) {
            float *h_m = (float *)calloc(n_in1 * n_out, sizeof(float));
            float *h_v = (float *)calloc(n_in1 * n_out, sizeof(float));
            for (int k = 0; k < n_in1; k++) {
                if (net->adamM[l][k])
                    memcpy(h_m + k * n_out, net->adamM[l][k], n_out * sizeof(float));
                if (net->adamV[l][k])
                    memcpy(h_v + k * n_out, net->adamV[l][k], n_out * sizeof(float));
            }
            cudaMemcpy(ctx->m_d[l], h_m, wsz, cudaMemcpyHostToDevice);
            cudaMemcpy(ctx->v_d[l], h_v, wsz, cudaMemcpyHostToDevice);
            free(h_m); free(h_v);
        } else {
            cudaMemset(ctx->m_d[l], 0, wsz);
            cudaMemset(ctx->v_d[l], 0, wsz);
        }

        cudaMalloc(&ctx->prev_d[l], wsz);  cudaMemset(ctx->prev_d[l], 0, wsz);
        cudaMalloc(&ctx->dw_d[l],   wsz);  cudaMemset(ctx->dw_d[l],   0, wsz);
    }

    /* ── activation / delta buffers (batch-sized) ── */
    int B = ctx->batch_size;
    ctx->z_d[0]     = NULL;
    ctx->delta_d[0] = NULL;
    /* a_d[0] will be set per-batch to point into inputs_d */
    ctx->a_d[0]     = NULL;

    for (int l = 1; l < L; l++) {
        int n_l   = net->neurons[l];
        int a_cols = (l < L - 1) ? n_l + 1 : n_l;   /* hidden has bias col */
        cudaMalloc(&ctx->z_d[l],     (size_t)B * n_l    * sizeof(float));
        cudaMalloc(&ctx->a_d[l],     (size_t)B * a_cols * sizeof(float));
        cudaMalloc(&ctx->delta_d[l], (size_t)B * n_l    * sizeof(float));
    }

    /* ── pre-scale and upload training data ── */
    int labelCols = fs->categoricalOutput ? 1 : nL;
    int inOff     = fs->groundTruthFirst  ? labelCols : 0;
    int outOff    = fs->groundTruthFirst  ? 0         : n0;

    float *h_inputs = (float *)malloc((size_t)N * n0p * sizeof(float));
    float *h_labels = (float *)malloc((size_t)N * nL  * sizeof(float));

    for (int i = 0; i < N; i++) {
        float *row = fs->inputRows[i].rowData;
        for (int j = 0; j < n0; j++) {
            float v  = row[inOff + j];
            float mn = net->minInputArray[j], mx = net->maxInputArray[j];
            h_inputs[i * n0p + j] = (mx == mn) ? 0.0f : (v - mn) / (mx - mn);
        }
        h_inputs[i * n0p + n0] = 1.0f;   /* bias */

        if (fs->categoricalOutput) {
            int lbl = (int)row[outOff];
            for (int j = 0; j < nL; j++)
                h_labels[i * nL + j] = (j == lbl) ? 1.0f : 0.0f;
        } else {
            for (int j = 0; j < nL; j++) {
                float v  = row[outOff + j];
                float mn = net->minOutputArray[j], mx = net->maxOutputArray[j];
                h_labels[i * nL + j] = (mx == mn) ? 0.0f : (v - mn) / (mx - mn);
            }
        }
    }

    cudaMalloc(&ctx->inputs_d, (size_t)N * n0p * sizeof(float));
    cudaMalloc(&ctx->labels_d, (size_t)N * nL  * sizeof(float));
    cudaMemcpy(ctx->inputs_d, h_inputs, (size_t)N * n0p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->labels_d, h_labels, (size_t)N * nL  * sizeof(float), cudaMemcpyHostToDevice);
    free(h_inputs);
    free(h_labels);

    /* ── stats scalars ── */
    cudaMalloc(&ctx->correct_d, sizeof(int));
    cudaMalloc(&ctx->sse_d,     sizeof(float));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA init error — falling back to CPU path");
        CudaFree(ctx, L);
        return NULL;
    }

    LOG_INFO("CUDA GPU context initialized");
    return ctx;
}

/* ── CudaFree ─────────────────────────────────────────────────────────── */

extern "C"
void CudaFree(void *opaque, int n_layers)
{
    if (!opaque) return;
    CudaContext *ctx = (CudaContext *)opaque;
    int L = n_layers;

    for (int l = 0; l < L - 1; l++) {
        if (ctx->w_d[l])    cudaFree(ctx->w_d[l]);
        if (ctx->m_d[l])    cudaFree(ctx->m_d[l]);
        if (ctx->v_d[l])    cudaFree(ctx->v_d[l]);
        if (ctx->prev_d[l]) cudaFree(ctx->prev_d[l]);
        if (ctx->dw_d[l])   cudaFree(ctx->dw_d[l]);
    }
    for (int l = 1; l < L; l++) {
        if (ctx->z_d[l])     cudaFree(ctx->z_d[l]);
        if (ctx->a_d[l])     cudaFree(ctx->a_d[l]);
        if (ctx->delta_d[l]) cudaFree(ctx->delta_d[l]);
    }
    if (ctx->inputs_d)  cudaFree(ctx->inputs_d);
    if (ctx->labels_d)  cudaFree(ctx->labels_d);
    if (ctx->correct_d) cudaFree(ctx->correct_d);
    if (ctx->sse_d)     cudaFree(ctx->sse_d);
    free(ctx);
}

/* ── CudaTrainEpoch ───────────────────────────────────────────────────── */

extern "C"
int CudaTrainEpoch(ThreadStruct *netPtr, void *opaque,
                   int *out_correct, float *out_sse)
{
    CudaContext *ctx = (CudaContext *)opaque;
    NetStruct   *net = netPtr->netStructPtr;
    int  L    = ctx->n_layers;
    int  N    = ctx->n_samples;
    int  B    = ctx->batch_size;
    int  n0p  = ctx->neurons[0] + 1;            /* input cols with bias  */
    int  nL   = ctx->neurons[L - 1];            /* output cols           */
    int  fn   = ctx->fn;

    ctx->epoch_count++;

    /* Reset epoch stats */
    cudaMemset(ctx->correct_d, 0, sizeof(int));
    cudaMemset(ctx->sse_d,     0, sizeof(float));

    for (int i = 0; i < N; i += B) {
        int batch = (i + B <= N) ? B : N - i;

        /* Point a_d[0] into the pre-uploaded dataset for this mini-batch */
        ctx->a_d[0] = ctx->inputs_d + (size_t)i * n0p;
        float *Y    = ctx->labels_d + (size_t)i * nL;

        /* ── FeedForward ── */
        for (int l = 1; l < L; l++) {
            int n_in1 = ctx->neurons[l - 1] + 1;
            int n_l   = ctx->neurons[l];
            int a_cols = (l < L - 1) ? n_l + 1 : n_l;

            /* z_d[l][batch × n_l] = a_d[l-1][batch × n_in1] @ w_d[l-1][n_in1 × n_l] */
            launch_gemm_nn(ctx->a_d[l - 1], n_in1,
                           ctx->w_d[l - 1], n_l,
                           ctx->z_d[l],     n_l,
                           batch, n_l, n_in1, 1.0f, 0.0f);

            /* a_d[l] = activate(z_d[l]); set bias col if not output layer */
            int total_a = batch * a_cols;
            activate_bias_k<<<(total_a + 255) / 256, 256>>>(
                ctx->z_d[l], n_l,
                ctx->a_d[l], a_cols,
                batch, n_l, (l < L - 1) ? 1 : 0, fn);

            /* apply inverted dropout to hidden layers during training */
            if (net->dropoutRate > 0.0f && l > 0 && l < L - 1) {
                int total_d = batch * n_l;
                unsigned long long seed = ctx->epoch_count * 7919ULL + (unsigned long long)l * 31ULL + (unsigned long long)(i / B);
                dropout_k<<<(total_d + 255) / 256, 256>>>(
                    ctx->a_d[l], a_cols, batch, n_l,
                    1.0f - net->dropoutRate, seed);
            }
        }

        /* ── Accuracy and SSE stats ── */
        stats_k<<<(batch + 255) / 256, 256>>>(
            ctx->a_d[L - 1], Y, batch, nL, ctx->correct_d, ctx->sse_d);

        /* ── BackProp ── */

        /* Output delta: delta_d[L-1] = (Y - a_d[L-1]) * deriv(a_d[L-1]) */
        output_delta_k<<<(batch * nL + 255) / 256, 256>>>(
            ctx->a_d[L - 1], Y, ctx->delta_d[L - 1], batch * nL, fn);

        /* Gradient for last weight layer: dw_d[L-2] += scale * a_d[L-2]^T @ delta_d[L-1] */
        {
            float scale   = 1.0f / (float)batch;
            int   n_in1   = ctx->neurons[L - 2] + 1;
            int   a_in_ld = n_in1;   /* a_d[L-2] stride = n_in1 (includes bias) */
            launch_gemm_tn(ctx->a_d[L - 2], a_in_ld,
                           ctx->delta_d[L - 1], nL,
                           ctx->dw_d[L - 2],   nL,
                           n_in1, nL, batch, scale, 1.0f);
        }

        /* Hidden layers (L-2 downto 1) */
        for (int l = L - 2; l >= 1; l--) {
            int n_l    = ctx->neurons[l];
            int n_next = ctx->neurons[l + 1];

            /*
             * delta_d[l][batch × n_l] = delta_d[l+1][batch × n_next]
             *                           @ w_d[l][:n_l, :]^T
             * W[l] is (n_l+1) × n_next; we use only first n_l rows → N = n_l.
             */
            launch_gemm_nt(ctx->delta_d[l + 1], n_next,
                           ctx->w_d[l],          n_next,
                           ctx->delta_d[l],      n_l,
                           batch, n_l, n_next, 1.0f, 0.0f);

            /* delta_d[l] *= deriv(a_d[l]).  With inverted dropout the activations
             * are scaled outside [0,1], so use a dedicated kernel that undoes
             * the scaling and zeros dropped neurons' deltas in one pass. */
            int a_stride_bp = n_l + 1;   /* a_d[l] has bias column */
            if (net->dropoutRate > 0.0f && l < L - 1) {
                apply_deriv_dropout_k<<<(batch * n_l + 255) / 256, 256>>>(
                    ctx->a_d[l], a_stride_bp,
                    ctx->delta_d[l], n_l,
                    batch, n_l, fn,
                    1.0f - net->dropoutRate);
            } else {
                apply_deriv_k<<<(batch * n_l + 255) / 256, 256>>>(
                    ctx->a_d[l], a_stride_bp,
                    ctx->delta_d[l], n_l,
                    batch, n_l, fn);
            }

            /* dw_d[l-1] += scale * a_d[l-1]^T @ delta_d[l] */
            float scale  = 1.0f / (float)batch;
            int n_in1_l  = ctx->neurons[l - 1] + 1;
            int a_in1_ld = n_in1_l;
            launch_gemm_tn(ctx->a_d[l - 1], a_in1_ld,
                           ctx->delta_d[l],  n_l,
                           ctx->dw_d[l - 1], n_l,
                           n_in1_l, n_l, batch, scale, 1.0f);
        }

        /* ── Weight update ── */
        ctx->adamT++;
        float b1  = net->adamBeta1,   b2  = net->adamBeta2;
        float bc1 = 1.0f - (float)powf(b1, (float)ctx->adamT);
        float bc2 = 1.0f - (float)powf(b2, (float)ctx->adamT);
        float lr  = net->learningRate;
        float eps = net->adamEpsilon;
        float dec = net->weightDecay;
        float mom = net->momentum;

        for (int l = 0; l < L - 1; l++) {
            int wsize = (ctx->neurons[l] + 1) * ctx->neurons[l + 1];
            int blks  = (wsize + 255) / 256;
            if (net->useAdam)
                adam_update_k<<<blks, 256>>>(
                    ctx->w_d[l], ctx->m_d[l], ctx->v_d[l], ctx->dw_d[l],
                    lr, b1, b2, eps, bc1, bc2, dec, wsize);
            else
                sgd_update_k<<<blks, 256>>>(
                    ctx->w_d[l], ctx->prev_d[l], ctx->dw_d[l],
                    lr, mom, dec, wsize);
        }
    }

    cudaDeviceSynchronize();

    /* Pull epoch stats back to host */
    cudaMemcpy(out_correct, ctx->correct_d, sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(out_sse,     ctx->sse_d,     sizeof(float), cudaMemcpyDeviceToHost);

    /* Update adamT in net so the CPU path (checkpoint saves) sees it */
    net->adamT = ctx->adamT;

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* ── CudaSyncWeightsToHost ────────────────────────────────────────────── */

extern "C"
void CudaSyncWeightsToHost(ThreadStruct *netPtr, void *opaque)
{
    if (!opaque) return;
    CudaContext *ctx = (CudaContext *)opaque;
    NetStruct   *net = netPtr->netStructPtr;
    int L = ctx->n_layers;

    for (int l = 0; l < L - 1; l++) {
        int n_in1 = net->neurons[l] + 1;
        int n_out = net->neurons[l + 1];
        int wsz   = n_in1 * n_out;
        float *h_w = (float *)malloc(wsz * sizeof(float));
        cudaMemcpy(h_w, ctx->w_d[l], (size_t)wsz * sizeof(float), cudaMemcpyDeviceToHost);
        unflatten_weights(net, l, h_w);
        free(h_w);
    }
}

/* ── CudaSyncWeightsToDevice ──────────────────────────────────────────── */

extern "C"
void CudaSyncWeightsToDevice(ThreadStruct *netPtr, void *opaque)
{
    if (!opaque) return;
    CudaContext *ctx = (CudaContext *)opaque;
    NetStruct   *net = netPtr->netStructPtr;
    int L = ctx->n_layers;

    for (int l = 0; l < L - 1; l++) {
        int n_in1 = net->neurons[l] + 1;
        int n_out = net->neurons[l + 1];
        int wsz   = n_in1 * n_out;
        float *h_w = flatten_weights(net, l);
        cudaMemcpy(ctx->w_d[l], h_w, (size_t)wsz * sizeof(float), cudaMemcpyHostToDevice);
        free(h_w);
    }
}
