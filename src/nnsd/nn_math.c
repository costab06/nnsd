/***************************************************************************
 *                          NNSD Math / Inference                           *
 **************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "log_utils.h"
#include "nn_globals.h"
#include "nn_math.h"
#include "packet.h"


/***************************** FeedForward  ****************************/
int
FeedForward (ThreadStruct * netPtr)
{
  int ccode = 0;

  char func[] = "FeedForward";
  ENTER (func);

  /* SAXPY form: k (input) outer, j (output) inner.
   * weights[layer][k] is a contiguous float row of length neurons[layer+1],
   * so the inner j loop accesses wrow[j] sequentially — AVX2 auto-vectorizable. */
  int i; for (i = 1; i < netPtr->netStructPtr->layers; i++) {
    int n_out = netPtr->netStructPtr->neurons[i];
    int n_in1 = netPtr->netStructPtr->neurons[i-1] + 1; /* +1 for BIAS */

    int j; for (j = 0; j < n_out; j++)
      netPtr->netStructPtr->vectors[i][j] = 0.0f;

    int k; for (k = 0; k < n_in1; k++) {
      float vk    = netPtr->netStructPtr->vectors[i-1][k];
      float *wrow = netPtr->netStructPtr->weights[i-1][k];
      for (j = 0; j < n_out; j++)
        netPtr->netStructPtr->vectors[i][j] += vk * wrow[j];
    }

    for (j = 0; j < n_out; j++)
      netPtr->netStructPtr->vectors[i][j] =
        nn_activate(netPtr->netStructPtr->vectors[i][j],
                    netPtr->netStructPtr->activationFunction);
  }

  EXIT (func);
  return ccode;
}


/******************************** BackProp *********************************/
int
BackProp (ThreadStruct * netPtr)
{
  float err;
  char func[] = "BackProp";
  ENTER (func);

  /* calculate the deltas for output layer */
  int i; for (i = 0; i < netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1]; i++) {
    err = netPtr->netStructPtr->expectedOutputArray[i] - netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers -1][i];
    netPtr->netStructPtr->deltas[netPtr->netStructPtr->layers - 1][i] = err *
      nn_deriv(netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][i],
               netPtr->netStructPtr->activationFunction);
  }

  /* calculate the deltas for hidden layers */
  for (i=netPtr->netStructPtr->layers-2;i>=0;i--) {
    int j; for (j=0;j<netPtr->netStructPtr->neurons[i]+1;j++) {
      netPtr->netStructPtr->deltas[i][j] = 0.0;
      int k; for(k=0;k<netPtr->netStructPtr->neurons[i+1];k++) {
        netPtr->netStructPtr->deltas[i][j]+=
          netPtr->netStructPtr->deltas[i+1][k] *
          netPtr->netStructPtr->weights[i][j][k];
      }
      netPtr->netStructPtr->deltas[i][j] *=
        nn_deriv(netPtr->netStructPtr->vectors[i][j],
                 netPtr->netStructPtr->activationFunction);
    }
  }

  /* update the weights */
  if (netPtr->netStructPtr->useAdam) {
    netPtr->netStructPtr->adamT++;
    float bc1 = 1.0f - (float)pow(netPtr->netStructPtr->adamBeta1, netPtr->netStructPtr->adamT);
    float bc2 = 1.0f - (float)pow(netPtr->netStructPtr->adamBeta2, netPtr->netStructPtr->adamT);
    float lr  = netPtr->netStructPtr->learningRate;
    for (i = 0; i < netPtr->netStructPtr->layers - 1; i++) {
      int j; for (j = 0; j < netPtr->netStructPtr->neurons[i+1]; j++) {
        int k; for (k = 0; k < netPtr->netStructPtr->neurons[i]+1; k++) {
          float g  = netPtr->netStructPtr->deltas[i+1][j]
                   * netPtr->netStructPtr->vectors[i][k];
          float m  = netPtr->netStructPtr->adamBeta1 * netPtr->netStructPtr->adamM[i][k][j]
                   + (1.0f - netPtr->netStructPtr->adamBeta1) * g;
          float v  = netPtr->netStructPtr->adamBeta2 * netPtr->netStructPtr->adamV[i][k][j]
                   + (1.0f - netPtr->netStructPtr->adamBeta2) * g * g;
          netPtr->netStructPtr->adamM[i][k][j] = m;
          netPtr->netStructPtr->adamV[i][k][j] = v;
          float dw = lr * (m / bc1) / (sqrtf(v / bc2) + netPtr->netStructPtr->adamEpsilon);
          netPtr->netStructPtr->weights[i][k][j] =
            netPtr->netStructPtr->weights[i][k][j] * (1.0f - netPtr->netStructPtr->weightDecay) + dw;
        }
      }
    }
  } else {
    for (i = 0; i < netPtr->netStructPtr->layers - 1; i++) {
      int j; for (j = 0; j < netPtr->netStructPtr->neurons[i+1]; j++) {
        int k; for (k = 0; k < netPtr->netStructPtr->neurons[i]+1; k++) {
          float deltaWeight = (netPtr->netStructPtr->learningRate *
                               netPtr->netStructPtr->deltas[i+1][j] *
                               netPtr->netStructPtr->vectors[i][k]) +
                              (netPtr->netStructPtr->momentum *
                               netPtr->netStructPtr->previousWeightChange[i][k][j]);
          netPtr->netStructPtr->previousWeightChange[i][k][j] = deltaWeight;
          netPtr->netStructPtr->weights[i][k][j] =
            netPtr->netStructPtr->weights[i][k][j] * (1.0f - netPtr->netStructPtr->weightDecay)
            + deltaWeight;
        }
      }
    }
  }

  EXIT(func);
  return 0;
}


/****************************** TestFact **********************************/
int
TestFact (ThreadStruct * netPtr, float *sumSquaredError)
{
  int i, factCorrect = TRUE;
  float difference, fabsDifference, sse=0.0;
  char func[] = "TestFact";
  ENTER (func);

  for (i = 0; i <
       netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1]; i++) {

    difference = netPtr->netStructPtr->expectedOutputArray[i] - netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers -1][i];
    sse += difference * difference;
    fabsDifference = fabs(difference);

    if (netPtr->networkType == TRAIN_PACKET || netPtr->networkType == RETRAIN_PACKET) {
      if (fabsDifference >  netPtr->netStructPtr->trainingTolerance) {
        factCorrect = FALSE;
        break;
      }
    } else {
      if (fabsDifference >  netPtr->netStructPtr->testingTolerance) {
        factCorrect = FALSE;
        break;
      }
    }
  }

  *sumSquaredError = sse;

  EXIT(func);
  return factCorrect;
}


/*********************************** ReadFact ***************************/
int
ReadFact (ThreadStruct * netPtr, int factNum)
{
  int ccode = 0;
  char func[] = "ReadFact";
  ENTER (func);

  int labelCols    = netPtr->fileStructPtr->categoricalOutput
                       ? 1
                       : netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1];
  int inputOffset  = netPtr->fileStructPtr->groundTruthFirst ? labelCols : 0;
  int outputOffset = netPtr->fileStructPtr->groundTruthFirst
                       ? 0
                       : netPtr->netStructPtr->neurons[0];

  int i; for (i=0;i<netPtr->netStructPtr->neurons[0];i++) {
    netPtr->netStructPtr->vectors[0][i] = netPtr->fileStructPtr->inputRows[factNum].rowData[inputOffset + i];
  }

  netPtr->netStructPtr->vectors[0][netPtr->netStructPtr->neurons[0]] = 1.0;

  if (netPtr->networkType == TRAIN_PACKET ||
      netPtr->networkType == TEST_PACKET ||
      netPtr->networkType == RETRAIN_PACKET) {
    if (netPtr->fileStructPtr->categoricalOutput) {
      int label  = (int)netPtr->fileStructPtr->inputRows[factNum].rowData[outputOffset];
      int numOut = netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1];
      for (i = 0; i < numOut; i++)
        netPtr->netStructPtr->expectedOutputArray[i] = (i == label) ? 1.0f : 0.0f;
    } else {
      for (i=0;i<netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers-1];i++) {
        netPtr->netStructPtr->expectedOutputArray[i] =
          netPtr->fileStructPtr->inputRows[factNum].rowData[outputOffset + i];
      }
    }
  }

  EXIT (func);
  return ccode;
}


/**************************** ScaleDownFact *********************************/
int
ScaleDownFact (float *v, int length, float *min, float *max)
{
  int i, ccode = 0;
  char func[] = "ScaleDownFact";
  ENTER (func);

  for (i = 0; i < length; i++)
    {
      if (max[i] == min[i])
        v[i] = 0.0f;
      else
        v[i] = (v[i] - min[i]) / (max[i] - min[i]);
    }

  EXIT (func);
  return ccode;
}


/**************************** ScaleUpFact *********************************/
int
ScaleUpFact (float *v, int length, float *min, float *max)
{
  float temp;
  int i, ccode = 0;
  char func[] = "ScaleUpFact";
  ENTER (func);

  for (i = 0; i < length; i++)
    {
      temp = v[i] * (max[i] - min[i]) + min[i];
      v[i] = temp;
    }

  EXIT(func);
  return ccode;
}


/****************************** NudgeWeights *****************************/
void
NudgeWeights (ThreadStruct * netPtr)
{
  int i, j, k;
  char func[] = "NudgeWeights";
  ENTER (func);

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));

  for (i = 0; i < netPtr->netStructPtr->layers - 1; i++)
    {
      for (j = 0; j < netPtr->netStructPtr->neurons[i]; j++)
        {
          for (k = 0; k < netPtr->netStructPtr->neurons[i+1]; k++)
            {
              if ((rand () % 10) < 5)
                netPtr->nudgeAmount *= -1;

              (netPtr->netStructPtr->weights[i][j][k]) +=
                ((netPtr->netStructPtr->weights[i][j][k]) *
                 netPtr->nudgeAmount);
            }
        }
    }

  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));
  EXIT(func);
}


/******************************** InitWtChange ******************************/
int
InitWtChange (ThreadStruct * netPtr)
{
  int i, j, k, ccode = 0;
  char func[] = "InitWtChange";
  ENTER (func);

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));

  for (i = 0; i < netPtr->netStructPtr->layers - 1; i++)
    for (j = 0; j < netPtr->netStructPtr->neurons[i + 1]; j++)
      for (k = 0; k < netPtr->netStructPtr->neurons[i]; k++)
        netPtr->netStructPtr->previousWeightChange[i][j][k] = INIT_WTCHANGE;

  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  EXIT(func);
  return ccode;
}


/***************************** SigmoidTransferFunction **************************/
int
SigmoidTransferFunction (float *v)
{
  int ccode = 0;

  if ((*v) < -8)
    *v = -8;

  if ((*v) > 8)
    *v = 8;

  float exp_value = exp(-(*v));
  *v = 1/(1+exp_value);

  return ccode;
}


int PrintVectors(ThreadStruct *netPtr) {
  char msg_buff[EIGHT_KB];
  int i; for (i = 0; i < netPtr->netStructPtr->layers; i++) {
    int j; for(j=0;j<netPtr->netStructPtr->neurons[i]+1;j++) {
      sprintf(msg_buff,"vector[%d][%d]=%f\n",i,j,netPtr->netStructPtr->vectors[i][j]);
    }
  }
  return SUCCESS;
}


int PrintWeights(ThreadStruct *netPtr) {
  char msg_buff[EIGHT_KB];
  int i; for (i = 0; i < netPtr->netStructPtr->layers -1; i++) {
    int j; for(j=0;j<netPtr->netStructPtr->neurons[i]+1;j++) {
      int k; for (k=0;k<netPtr->netStructPtr->neurons[i+1];k++) {
        sprintf(msg_buff,"weights[%d][%d][%d]=%f\n",i,j,k,netPtr->netStructPtr->weights[i][j][k]);
      }
    }
  }
  return SUCCESS;
}
