/***************************************************************************
 *                     NNSD CUDA GPU Training Interface                     *
 *                                                                           *
 * Pure C interface — no CUDA headers included here so this can be safely   *
 * included from ordinary .c files.  The CudaContext struct is defined only  *
 * inside nn_cuda.cu; callers hold it as a void*.                           *
 ***************************************************************************/
#ifndef NN_CUDA_H
#define NN_CUDA_H

#ifdef USE_CUDA

#include "nnsd.h"

/*
 * CudaInit — allocate device memory, upload weights and pre-scaled training
 * data.  Returns NULL if no CUDA device is present or any allocation fails.
 * Call after LoadAndValidateInputFile() and the min/max pass have finished.
 */
void *CudaInit(ThreadStruct *netPtr);

/*
 * CudaFree — release all device memory and destroy the context.
 */
void CudaFree(void *ctx, int n_layers);

/*
 * CudaTrainEpoch — run one full pass over the training set on the GPU.
 * Writes the total correct-prediction count into *out_correct and the total
 * sum-of-squared-errors into *out_sse.  Returns 0 on success.
 */
int CudaTrainEpoch(ThreadStruct *netPtr, void *ctx,
                   int *out_correct, float *out_sse);

/*
 * CudaSyncWeightsToHost — copy device weights → host NetStruct.
 * Must be called before WriteMatrixFile() or NudgeWeights().
 */
void CudaSyncWeightsToHost(ThreadStruct *netPtr, void *ctx);

/*
 * CudaSyncWeightsToDevice — copy host NetStruct weights → device.
 * Must be called after NudgeWeights() to push the nudged values to GPU.
 */
void CudaSyncWeightsToDevice(ThreadStruct *netPtr, void *ctx);

#endif /* USE_CUDA */

#endif /* NN_CUDA_H */
