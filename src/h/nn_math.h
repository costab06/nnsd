#ifndef NN_MATH_H
#define NN_MATH_H

#include <math.h>
#include "nnsd.h"
#include "common.h"

#define INIT_WTCHANGE .1

/* Activation functions — inline so both nn_math.c and nn_training.c can use
   them in hot paths without a function-call overhead. */
static inline float nn_activate(float z, int fn) {
  switch (fn) {
  case ACTIVATION_TANH:
    return tanhf(z);
  case ACTIVATION_RELU:
    return z > 0.0f ? z : 0.0f;
  case ACTIVATION_LEAKY_RELU:
    return z > 0.0f ? z : 0.01f * z;
  default: /* ACTIVATION_SIGMOID */
    if (z < -9.0f) z = -8.0f; else if (z > 9.0f) z = 8.0f;
    return 1.0f / (1.0f + expf(-z));
  }
}

static inline float nn_deriv(float a, int fn) {
  switch (fn) {
  case ACTIVATION_TANH:
    return 1.0f - a * a;
  case ACTIVATION_RELU:
    return a > 0.0f ? 1.0f : 0.0f;
  case ACTIVATION_LEAKY_RELU:
    return a >= 0.0f ? 1.0f : 0.01f;
  default: /* ACTIVATION_SIGMOID */
    return a * (1.0f - a);
  }
}

int FeedForward(ThreadStruct *netPtr);
int BackProp(ThreadStruct *netPtr);
int TestFact(ThreadStruct *netPtr, float *sumSquaredError);
int ReadFact(ThreadStruct *netPtr, int factNum);
int ScaleDownFact(float *v, int length, float *min, float *max);
int ScaleUpFact(float *v, int length, float *min, float *max);
void NudgeWeights(ThreadStruct *netPtr);
int InitWtChange(ThreadStruct *netPtr);
int SigmoidTransferFunction(float *v);
int PrintVectors(ThreadStruct *netPtr);
int PrintWeights(ThreadStruct *netPtr);

#endif /* NN_MATH_H */
