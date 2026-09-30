/***************************************************************************
 *                      NNSD Training / Run / Test Threads                  *
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "log_utils.h"
#include "nn_globals.h"
#include "nn_math.h"
#include "nn_fileio.h"
#include "nn_network.h"
#include "nn_packet.h"
#include "nn_training.h"
#ifdef USE_CUDA
#include "nn_cuda.h"
#endif


/* ── AllocGradAccum / FreeGradAccum: shared gradAccum on NetStruct ── */

static void AllocGradAccum(ThreadStruct *netPtr) {
  NetStruct *net = netPtr->netStructPtr;
  int i, j;
  for (i = 0; i < net->layers - 1; i++) {
    net->gradAccum[i] = calloc(net->neurons[i] + 1, sizeof(float *));
    for (j = 0; j < net->neurons[i] + 1; j++)
      net->gradAccum[i][j] = calloc(net->neurons[i + 1], sizeof(float));
  }
}

static void FreeGradAccum(ThreadStruct *netPtr) {
  NetStruct *net = netPtr->netStructPtr;
  int i, j;
  if (net == NULL || net->gradAccum[0] == NULL) return;
  for (i = 0; i < net->layers - 1; i++) {
    if (net->gradAccum[i] == NULL) continue;
    for (j = 0; j < net->neurons[i] + 1; j++)
      if (net->gradAccum[i][j]) { free(net->gradAccum[i][j]); net->gradAccum[i][j] = NULL; }
    free(net->gradAccum[i]);
    net->gradAccum[i] = NULL;
  }
}


/* ── ApplyBatchGradients: sum per-worker grads, apply with momentum ── */

static void ApplyBatchGradients(ThreadStruct *netPtr,
                                PoolWorker *pool, int poolSize,
                                int failedInBatch) {
  NetStruct *net = netPtr->netStructPtr;
  int i, j, k, b;
  if (failedInBatch == 0) return;

  for (b = 0; b < poolSize; b++) {
    if (pool[b].factCorrect) continue;
    for (i = 0; i < net->layers - 1; i++)
      for (k = 0; k < net->neurons[i] + 1; k++)
        for (j = 0; j < net->neurons[i + 1]; j++)
          net->gradAccum[i][k][j] += pool[b].localGrad[i][k][j];
  }

  float scale = 1.0f / (float)failedInBatch;

  if (net->useAdam) {
    net->adamT++;
    float bc1 = 1.0f - (float)pow(net->adamBeta1, net->adamT);
    float bc2 = 1.0f - (float)pow(net->adamBeta2, net->adamT);
    float lr  = net->learningRate;
    for (i = 0; i < net->layers - 1; i++)
      for (k = 0; k < net->neurons[i] + 1; k++)
        for (j = 0; j < net->neurons[i + 1]; j++) {
          float g  = net->gradAccum[i][k][j] * scale;
          float m  = net->adamBeta1 * net->adamM[i][k][j] + (1.0f - net->adamBeta1) * g;
          float v  = net->adamBeta2 * net->adamV[i][k][j] + (1.0f - net->adamBeta2) * g * g;
          net->adamM[i][k][j] = m;
          net->adamV[i][k][j] = v;
          float mhat = m / bc1;
          float vhat = v / bc2;
          float dw   = lr * mhat / (sqrtf(vhat) + net->adamEpsilon);
          net->weights[i][k][j] = net->weights[i][k][j] * (1.0f - net->weightDecay) + dw;
          net->gradAccum[i][k][j] = 0.0f;
        }
  } else {
    for (i = 0; i < net->layers - 1; i++)
      for (j = 0; j < net->neurons[i + 1]; j++)
        for (k = 0; k < net->neurons[i] + 1; k++) {
          float dw = net->gradAccum[i][k][j] * scale
                   + net->momentum * net->previousWeightChange[i][k][j];
          net->previousWeightChange[i][k][j] = dw;
          net->weights[i][k][j] = net->weights[i][k][j] * (1.0f - net->weightDecay) + dw;
          net->gradAccum[i][k][j] = 0.0f;
        }
  }
}


/* ── PoolWorkerThread: fully inlined, zero mutex calls in hot path ── */

static void *PoolWorkerThread(void *arg) {
  PoolWorker   *w      = (PoolWorker *)arg;
  ThreadStruct *netPtr = w->netPtr;
  NetStruct    *net    = netPtr->netStructPtr;
  int i, layer, j, k;

  while (1) {
    sem_wait(&w->work_sem);
    if (w->factIndex < 0) { sem_post(&w->done_sem); return NULL; }

    int fIdx = w->factIndex;

    /* 1. Read inputs into thread-local vectors[0] */
    int labelCols   = netPtr->fileStructPtr->categoricalOutput
                        ? 1 : net->neurons[net->layers - 1];
    int inputOffset  = netPtr->fileStructPtr->groundTruthFirst ? labelCols : 0;
    int outputOffset = netPtr->fileStructPtr->groundTruthFirst ? 0 : net->neurons[0];

    for (i = 0; i < net->neurons[0]; i++)
      w->vectors[0][i] =
        netPtr->fileStructPtr->inputRows[fIdx].rowData[inputOffset + i];
    w->vectors[0][net->neurons[0]] = 1.0f;   /* BIAS */

    /* 2. Read expected output (one-hot if categorical) */
    int numOut = net->neurons[net->layers - 1];
    if (netPtr->fileStructPtr->categoricalOutput) {
      int label = (int)netPtr->fileStructPtr->inputRows[fIdx].rowData[outputOffset];
      for (i = 0; i < numOut; i++)
        w->expectedOutput[i] = (i == label) ? 1.0f : 0.0f;
    } else {
      for (i = 0; i < numOut; i++)
        w->expectedOutput[i] =
          netPtr->fileStructPtr->inputRows[fIdx].rowData[outputOffset + i];
    }

    /* 3. ScaleDown (inlined) */
    for (i = 0; i < net->neurons[0]; i++) {
      float mn = net->minInputArray[i], mx = net->maxInputArray[i];
      w->vectors[0][i] = (mx == mn) ? 0.0f : (w->vectors[0][i] - mn) / (mx - mn);
    }
    for (i = 0; i < numOut; i++) {
      float mn = net->minOutputArray[i], mx = net->maxOutputArray[i];
      w->expectedOutput[i] = (mx == mn) ? 0.0f : (w->expectedOutput[i] - mn) / (mx - mn);
    }

    /* 4. FeedForward (inlined SAXPY) */
    for (layer = 1; layer < net->layers; layer++) {
      int n_out = net->neurons[layer];
      int n_in1 = net->neurons[layer - 1] + 1;
      for (j = 0; j < n_out; j++) w->vectors[layer][j] = 0.0f;
      for (k = 0; k < n_in1; k++) {
        float vk    = w->vectors[layer - 1][k];
        float *wrow = net->weights[layer - 1][k];
        for (j = 0; j < n_out; j++)
          w->vectors[layer][j] += vk * wrow[j];
      }
      for (j = 0; j < n_out; j++)
        w->vectors[layer][j] = nn_activate(w->vectors[layer][j], net->activationFunction);
      /* Inverted dropout on hidden layers only */
      if (net->dropoutRate > 0.0f && layer > 0 && layer < net->layers - 1) {
        float keep = 1.0f - net->dropoutRate;
        for (j = 0; j < n_out; j++) {
          float r = (float)rand_r(&w->rng_state) / (float)RAND_MAX;
          if (r < net->dropoutRate) {
            w->vectors[layer][j]     = 0.0f;
            w->dropout_mask[layer][j] = 0.0f;
          } else {
            w->vectors[layer][j]     /= keep;
            w->dropout_mask[layer][j] = 1.0f;
          }
        }
      }
    }

    /* 5. TestFact */
    float sse = 0.0f;
    for (i = 0; i < numOut; i++) {
      float diff = w->expectedOutput[i] - w->vectors[net->layers - 1][i];
      sse += diff * diff;
    }
    w->sumSquaredError = sse;

    w->factCorrect = 1;
    for (i = 0; i < numOut; i++) {
      if (fabsf(w->expectedOutput[i] - w->vectors[net->layers - 1][i]) > net->trainingTolerance) {
        w->factCorrect = 0; break;
      }
    }

    if (netPtr->fileStructPtr->categoricalOutput) {
      int predClass, expClass;
      if (numOut == 1) {
        /* Threshold comparison for single-output binary — argmax is degenerate. */
        predClass = (w->vectors[net->layers - 1][0] >= 0.5f) ? 1 : 0;
        expClass  = (w->expectedOutput[0]            >= 0.5f) ? 1 : 0;
      } else {
        predClass = 0; expClass = 0;
        for (i = 1; i < numOut; i++) {
          if (w->vectors[net->layers - 1][i] > w->vectors[net->layers - 1][predClass]) predClass = i;
          if (w->expectedOutput[i]            > w->expectedOutput[expClass])            expClass  = i;
        }
      }
      w->factPredictedCorrect = (predClass == expClass) ? 1 : 0;
    } else {
      w->factPredictedCorrect = w->factCorrect;
    }

    if (!w->factCorrect) {
      /* 6. Compute output deltas */
      for (i = 0; i < numOut; i++) {
        float err = w->expectedOutput[i] - w->vectors[net->layers - 1][i];
        w->deltas[net->layers - 1][i] =
          err * nn_deriv(w->vectors[net->layers - 1][i], net->activationFunction);
      }
      /* 7. Backpropagate deltas through hidden layers */
      for (layer = net->layers - 2; layer >= 0; layer--) {
        for (j = 0; j < net->neurons[layer] + 1; j++) {
          float d = 0.0f;
          for (k = 0; k < net->neurons[layer + 1]; k++)
            d += w->deltas[layer + 1][k] * net->weights[layer][j][k];
          /* Undo inverted-dropout scaling before computing the derivative so
           * that sigmoid's a*(1-a) is not evaluated on a value outside [0,1]. */
          float a_for_deriv = w->vectors[layer][j];
          if (net->dropoutRate > 0.0f && layer > 0 && layer < net->layers - 1
              && w->dropout_mask[layer][j] > 0.0f)
            a_for_deriv *= (1.0f - net->dropoutRate);
          w->deltas[layer][j] =
            d * nn_deriv(a_for_deriv, net->activationFunction);
          /* Zero delta for dropped neurons */
          if (net->dropoutRate > 0.0f && layer > 0 && layer < net->layers - 1)
            w->deltas[layer][j] *= w->dropout_mask[layer][j];
        }
      }
      /* 8. Accumulate into per-worker localGrad */
      for (layer = 0; layer < net->layers - 1; layer++) {
        int n_out  = net->neurons[layer + 1];
        int n_in1  = net->neurons[layer] + 1;
        float *delta_next = w->deltas[layer + 1];
        for (k = 0; k < n_in1; k++) {
          float vk       = w->vectors[layer][k];
          float *grad_row = w->localGrad[layer][k];
          for (j = 0; j < n_out; j++)
            grad_row[j] += net->learningRate * delta_next[j] * vk;
        }
      }
    }

    sem_post(&w->done_sem);
  }
  return NULL;
}


/************************** RunNetwork ***********************************/
void *
RunNetwork (void * arg)
{
  ThreadStruct * netPtr = (ThreadStruct *) arg;
  int i;
  char func[] = "RunNetwork";
  ENTER (func);

  netPtr->netStructPtr->trainingStartTime = time(NULL);

  WriteOutputFileHeader(netPtr,MODE_RUNNING);

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));
  netPtr->netStructPtr->meanSumSquaredError = 0.0;
  netPtr->netStructPtr->percentFactsCorrect = 0.0;
  netPtr->netStructPtr->numberOfPasses = 0;
  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  for (i=0;i<netPtr->fileStructPtr->numInputRows && GetNetStatusFlag(netPtr) == STATUS_RUN; i++) {
    ReadFact (netPtr,i);

    ScaleDownFact (netPtr->netStructPtr->vectors[0],
                   netPtr->netStructPtr->neurons[0],
                   netPtr->netStructPtr->minInputArray,
                   netPtr->netStructPtr->maxInputArray);

    ScaleDownFact (netPtr->netStructPtr->expectedOutputArray,
                   netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],
                   netPtr->netStructPtr->minOutputArray,
                   netPtr->netStructPtr->maxOutputArray);

    FeedForward (netPtr);

    ScaleUpFact (netPtr->netStructPtr->vectors[0],
                 netPtr->netStructPtr->neurons[0],
                 netPtr->netStructPtr->minInputArray,
                 netPtr->netStructPtr->maxInputArray);

    ScaleUpFact (netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1],
                 netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],
                 netPtr->netStructPtr->minOutputArray,
                 netPtr->netStructPtr->maxOutputArray);

    WriteOutputFileLine(netPtr,MODE_RUNNING);
  }

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));
  netPtr->netStructPtr->numberOfPasses++;
  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  WriteStatsFileLine(netPtr, netPtr->netStructPtr->numberOfPasses, 0, 0.0, 0);

  SendStatsPacket(netPtr);
  SendNetFinishedPacket(netPtr);

  pthread_detach(netPtr->thread);
  RemoveNetwork (netPtr);

  EXIT (func);
  return NULL;
}


/************************** TrainNetwork ***********************************/
void *
TrainNetwork (void *arg) {
  ThreadStruct *netPtr = (ThreadStruct *) arg;

  int i, ccode = 0, failedFacts = 1;
  int correctFacts = 0;
  float sumSquaredError = 0.0;
  time_t s1, t1;
  char msg_buff[EIGHT_KB];
  char func[] = "TrainNetwork";

  ENTER (func);

  s1 = time (NULL);
  t1 = time (NULL);

  WriteStatsFileHeader(netPtr);

  if (writeTrainingFailedFacts == TRUE) {
    WriteFailedFactsFileHeader(netPtr);
  }

  WriteOutputFileHeader(netPtr,MODE_TRAINING);

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));
  netPtr->netStructPtr->meanSumSquaredError = 0.0;
  netPtr->netStructPtr->percentFactsCorrect = 0.0;
  netPtr->netStructPtr->numberOfPasses = 0;
  netPtr->netStructPtr->trainingStartTime = time(NULL);
  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  /* One-line start record: easy to grep, contains everything needed to triage. */
  sprintf(msg_buff,
          "TRAIN START  netID=%d arch=%d-%d-%d-%d-%d-%d-%d-%d layers=%d "
          "file=%s tsp=%.1f timeout=%lds dropout=%.2f adam=%d",
          netPtr->netID,
          netPtr->netStructPtr->neurons[0], netPtr->netStructPtr->neurons[1],
          netPtr->netStructPtr->neurons[2], netPtr->netStructPtr->neurons[3],
          netPtr->netStructPtr->neurons[4], netPtr->netStructPtr->neurons[5],
          netPtr->netStructPtr->neurons[6], netPtr->netStructPtr->neurons[7],
          netPtr->netStructPtr->layers,
          netPtr->fileStructPtr->infile,
          netPtr->netStructPtr->trainingSuccessPercentage,
          netPtr->netStructPtr->trainingTimeout,
          netPtr->netStructPtr->dropoutRate,
          netPtr->netStructPtr->useAdam);
  LOG_INFO(msg_buff);

  SendStatsPacket(netPtr);

#ifdef USE_CUDA
  if (netPtr->netStructPtr->cudaCtx != NULL) {
    /* ── GPU training path ─────────────────────────────────────────── */
    void *ctx = netPtr->netStructPtr->cudaCtx;

    while (netPtr->netStructPtr->percentFactsCorrect
             < netPtr->netStructPtr->trainingSuccessPercentage
           && GetNetStatusFlag(netPtr) == STATUS_RUN)
      {
        if (netPtr->netStructPtr->trainingTimeout > 0) {
          long elapsed = (long)difftime(time(NULL),
                                        netPtr->netStructPtr->trainingStartTime);
          if (elapsed >= netPtr->netStructPtr->trainingTimeout) {
            sprintf(msg_buff,
                    "Net %d: training timeout reached (%lds), stopping.",
                    netPtr->netID, netPtr->netStructPtr->trainingTimeout);
            LOG_INFO(msg_buff);
            break;
          }
        }

        /* Nudge: sync to host, perturb, push back to device */
        if (netPtr->nudgeAmount != 0.0) {
          CudaSyncWeightsToHost(netPtr, ctx);
          NudgeWeights(netPtr);
          CudaSyncWeightsToDevice(netPtr, ctx);
          netPtr->nudgeAmount = 0.0;
        }

        int   gpu_correct = 0;
        float gpu_sse     = 0.0f;
        if (CudaTrainEpoch(netPtr, ctx, &gpu_correct, &gpu_sse) != 0) {
          LOG_ERROR("CudaTrainEpoch reported a CUDA error — check GPU state");
        }

        pthread_mutex_lock(&(netPtr->threadInfoHoldMtx));
        netPtr->netStructPtr->numberOfPasses++;
        netPtr->netStructPtr->meanSumSquaredError =
          gpu_sse / (float)netPtr->fileStructPtr->numInputRows;
        netPtr->netStructPtr->percentFactsCorrect =
          100.0f * (float)gpu_correct / (float)netPtr->fileStructPtr->numInputRows;
        pthread_mutex_unlock(&(netPtr->threadInfoHoldMtx));

        if (debugWeights) {
          CudaSyncWeightsToHost(netPtr, ctx);
          float w00 = netPtr->netStructPtr->weights[0][0][0];
          float w10 = netPtr->netStructPtr->weights[1][0][0];
          sprintf(msg_buff, "pass=%d correct=%d/%.1f%% w[0][0][0]=%.9f w[1][0][0]=%.9f",
                  netPtr->netStructPtr->numberOfPasses, gpu_correct,
                  netPtr->netStructPtr->percentFactsCorrect, w00, w10);
          LOG_DEBUG_WEIGHTS(msg_buff);
        }

        WriteStatsFileLine(netPtr,
                           netPtr->netStructPtr->numberOfPasses,
                           (netPtr->netStructPtr->percentFactsCorrect
                              >= netPtr->netStructPtr->trainingSuccessPercentage) ? 1 : 0,
                           netPtr->netStructPtr->meanSumSquaredError,
                           netPtr->netStructPtr->percentFactsCorrect);

        if ((unsigned int)difftime(time(NULL), t1) > FOUR_SECONDS) {
          SendStatsPacket(netPtr);
          t1 = time(NULL);
        }

        if ((unsigned int)difftime(time(NULL), s1) > SIXTY_SECONDS) {
          CudaSyncWeightsToHost(netPtr, ctx);
          if ((ccode = WriteMatrixFile(netPtr))) {
            sprintf(msg_buff, "Error writing matrix file, ccode: %d", ccode);
            LOG_ERROR(msg_buff);
          }
          s1 = time(NULL);
        }
      }

    /* Final weight sync and save */
    CudaSyncWeightsToHost(netPtr, ctx);
    goto gpu_write_matrix;
  }
#endif

  AllocGradAccum(netPtr);

  int batchSize = (int)sysconf(_SC_NPROCESSORS_ONLN);
  if (batchSize < 1) batchSize = 1;

  PoolWorker *pool = calloc(batchSize, sizeof(PoolWorker));
  {
    int b, layer, jj;
    for (b = 0; b < batchSize; b++) {
      PoolWorker *pw = &pool[b];
      pw->netPtr = netPtr;
      sem_init(&pw->work_sem, 0, 0);
      sem_init(&pw->done_sem, 0, 0);
      for (layer = 0; layer < netPtr->netStructPtr->layers; layer++) {
        pw->vectors[layer] =
          calloc(netPtr->netStructPtr->neurons[layer] + 1, sizeof(float));
        pw->deltas[layer] =
          calloc(netPtr->netStructPtr->neurons[layer] + 1, sizeof(float));
      }
      pw->expectedOutput =
        calloc(netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],
               sizeof(float));
      pw->rng_state = (unsigned int)(b + 1) * 1234567u
                    + (unsigned int)time(NULL);
      for (layer = 0; layer < netPtr->netStructPtr->layers; layer++)
        pw->dropout_mask[layer] =
          calloc(netPtr->netStructPtr->neurons[layer] + 1, sizeof(float));
      for (layer = 0; layer < netPtr->netStructPtr->layers - 1; layer++) {
        pw->localGrad[layer] =
          calloc(netPtr->netStructPtr->neurons[layer] + 1, sizeof(float *));
        for (jj = 0; jj < netPtr->netStructPtr->neurons[layer] + 1; jj++)
          pw->localGrad[layer][jj] =
            calloc(netPtr->netStructPtr->neurons[layer + 1], sizeof(float));
      }
      pthread_create(&pw->thread, NULL, PoolWorkerThread, pw);
    }
  }

  while (netPtr->netStructPtr->percentFactsCorrect < netPtr->netStructPtr->trainingSuccessPercentage
         && GetNetStatusFlag(netPtr) == STATUS_RUN)
    {
      if (netPtr->netStructPtr->trainingTimeout > 0) {
        long elapsed = (long)difftime(time(NULL), netPtr->netStructPtr->trainingStartTime);
        if (elapsed >= netPtr->netStructPtr->trainingTimeout) {
          sprintf(msg_buff, "Net %d: training timeout reached (%lds), stopping.",
                  netPtr->netID, netPtr->netStructPtr->trainingTimeout);
          LOG_INFO(msg_buff);
          break;
        }
      }

      if (netPtr->nudgeAmount != 0.0)
        {
          NudgeWeights (netPtr);
          netPtr->nudgeAmount = 0.0;
        }

      failedFacts = 0;
      correctFacts = 0;
      sumSquaredError = 0.0;

      for (i = 0;
           i < netPtr->fileStructPtr->numInputRows && GetNetStatusFlag(netPtr) == STATUS_RUN;
           i += batchSize) {

        int thisBatch = batchSize;
        if (i + thisBatch > netPtr->fileStructPtr->numInputRows)
          thisBatch = netPtr->fileStructPtr->numInputRows - i;

        int b;
        for (b = 0; b < thisBatch; b++) {
          pool[b].factIndex       = i + b;
          pool[b].factCorrect     = 1;
          pool[b].sumSquaredError = 0.0f;
          {
            int layer, jj;
            for (layer = 0; layer < netPtr->netStructPtr->layers - 1; layer++)
              for (jj = 0; jj < netPtr->netStructPtr->neurons[layer] + 1; jj++)
                memset(pool[b].localGrad[layer][jj], 0,
                       netPtr->netStructPtr->neurons[layer + 1] * sizeof(float));
          }
          sem_post(&pool[b].work_sem);
        }

        int failedInBatch = 0;
        for (b = 0; b < thisBatch; b++) {
          sem_wait(&pool[b].done_sem);
          sumSquaredError += pool[b].sumSquaredError;
          if (!pool[b].factCorrect) {
            failedFacts++;
            failedInBatch++;
          }
          if (pool[b].factPredictedCorrect)
            correctFacts++;
        }

        if (failedInBatch > 0)
          ApplyBatchGradients(netPtr, pool, thisBatch, failedInBatch);

      } /* end mini-batch loop */

      if (debugWeights) {
        float w00 = netPtr->netStructPtr->weights[0][0][0];
        float w10 = netPtr->netStructPtr->weights[1][0][0];
        sprintf(msg_buff, "pass=%d failedFacts=%d w[0][0][0]=%.9f w[1][0][0]=%.9f",
                netPtr->netStructPtr->numberOfPasses,
                failedFacts, w00, w10);
        LOG_DEBUG_WEIGHTS(msg_buff);
      }

      int passComplete = (i >= netPtr->fileStructPtr->numInputRows);
      pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));
      if (passComplete) {
        netPtr->netStructPtr->numberOfPasses++;
        netPtr->netStructPtr->meanSumSquaredError = (float)sumSquaredError/(float)netPtr->fileStructPtr->numInputRows;
        netPtr->netStructPtr->percentFactsCorrect = 100.0f * (float)correctFacts /
          (float)(netPtr->fileStructPtr->numInputRows);
      }
      pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

      WriteStatsFileLine(netPtr,
                         netPtr->netStructPtr->numberOfPasses, (failedFacts>0)?0:1,
                         netPtr->netStructPtr->meanSumSquaredError,
                         netPtr->netStructPtr->percentFactsCorrect);

      if (failedFacts==0) {
        SendStatsPacket(netPtr);
      } else {
        if ((unsigned int) difftime (time (NULL), t1) > FOUR_SECONDS) {
          SendStatsPacket(netPtr);
          t1 = time(NULL);
        }
      }

      if ((unsigned int) difftime (time (NULL), s1) > SIXTY_SECONDS) {
        if ((ccode = WriteMatrixFile (netPtr)))
          {
            sprintf(msg_buff,"Error writing matrix file, ccode: %d",ccode);
            LOG_ERROR(msg_buff);
          }
        s1 = time (NULL);
      }

    }

  /* Shutdown pool workers and free resources */
  {
    int b, layer, jj;
    for (b = 0; b < batchSize; b++) {
      pool[b].factIndex = -1;
      sem_post(&pool[b].work_sem);
      sem_wait(&pool[b].done_sem);
      pthread_join(pool[b].thread, NULL);
      sem_destroy(&pool[b].work_sem);
      sem_destroy(&pool[b].done_sem);
      for (layer = 0; layer < netPtr->netStructPtr->layers; layer++) {
        free(pool[b].vectors[layer]);
        free(pool[b].deltas[layer]);
        free(pool[b].dropout_mask[layer]);
      }
      free(pool[b].expectedOutput);
      for (layer = 0; layer < netPtr->netStructPtr->layers - 1; layer++) {
        for (jj = 0; jj < netPtr->netStructPtr->neurons[layer] + 1; jj++)
          free(pool[b].localGrad[layer][jj]);
        free(pool[b].localGrad[layer]);
      }
    }
    free(pool);
  }
  FreeGradAccum(netPtr);

#ifdef USE_CUDA
gpu_write_matrix:
#endif
  ccode = WriteMatrixFile (netPtr);
  if (ccode) {
    sprintf(msg_buff,"Error writing network, ccode: %d",ccode);
    LOG_ERROR(msg_buff);
  }

  ccode = BackupMatrixFile (netPtr);
  if (ccode) {
    sprintf(msg_buff,"Error backing up matrix file for network, ccode: %d",ccode);
    LOG_ERROR(msg_buff);
  }

  {
    long elapsed = (long)difftime(time(NULL), netPtr->netStructPtr->trainingStartTime);
    sprintf(msg_buff,
            "TRAIN FINISH netID=%d passes=%d mse=%.6f pct=%.2f elapsed=%lds file=%s",
            netPtr->netID,
            netPtr->netStructPtr->numberOfPasses,
            netPtr->netStructPtr->meanSumSquaredError,
            netPtr->netStructPtr->percentFactsCorrect,
            elapsed,
            netPtr->fileStructPtr->infile);
    LOG_INFO(msg_buff);
  }

  SendNetFinishedPacket(netPtr);

  pthread_detach(netPtr->thread);
  RemoveNetwork (netPtr);

  EXIT (func);
  return NULL;
}


/************************** TestNetwork ***********************************/
void *
TestNetwork (void *arg) {
  ThreadStruct *netPtr = (ThreadStruct *) arg;

  int i, failedFacts = 0;
  float sumSquaredError = 0.0;
  float sumSquaredErrorThisFact = 0.0;
  float sumAbsErrorOrig = 0.0f;
  float sumSqErrorOrig  = 0.0f;
  char func[] = "TestNetwork";
  ENTER (func);

  netPtr->netStructPtr->trainingStartTime = time(NULL);

  SendStatsPacket(netPtr);

  WriteStatsFileHeader(netPtr);
  WriteFailedFactsFileHeader(netPtr);
  WriteOutputFileHeader(netPtr,MODE_TESTING);

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));
  netPtr->netStructPtr->meanSumSquaredError = 0.0;
  netPtr->netStructPtr->percentFactsCorrect = 0.0;
  netPtr->netStructPtr->numberOfPasses = 0;
  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  for (i=0;i<netPtr->fileStructPtr->numInputRows && GetNetStatusFlag(netPtr) == STATUS_RUN; i++) {
    ReadFact (netPtr,i);

    ScaleDownFact (netPtr->netStructPtr->vectors[0],
                   netPtr->netStructPtr->neurons[0],
                   netPtr->netStructPtr->minInputArray,
                   netPtr->netStructPtr->maxInputArray);

    ScaleDownFact (netPtr->netStructPtr->expectedOutputArray,
                   netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],
                   netPtr->netStructPtr->minOutputArray,
                   netPtr->netStructPtr->maxOutputArray);

    FeedForward (netPtr);

    int factCorrect = TestFact (netPtr,&sumSquaredErrorThisFact);
    sumSquaredError+=sumSquaredErrorThisFact;

    if (netPtr->fileStructPtr->categoricalOutput) {
      int numOut = netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1];
      int predClass, expClass;
      if (numOut == 1) {
        predClass = (netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][0] >= 0.5f) ? 1 : 0;
        expClass  = (netPtr->netStructPtr->expectedOutputArray[0] >= 0.5f) ? 1 : 0;
      } else {
        predClass = 0; expClass = 0;
        int ai;
        for (ai = 1; ai < numOut; ai++) {
          if (netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][ai] >
              netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][predClass])
            predClass = ai;
          if (netPtr->netStructPtr->expectedOutputArray[ai] >
              netPtr->netStructPtr->expectedOutputArray[expClass])
            expClass = ai;
        }
      }
      factCorrect = (predClass == expClass) ? 1 : 0;
    }

    ScaleUpFact (netPtr->netStructPtr->vectors[0],
                 netPtr->netStructPtr->neurons[0],
                 netPtr->netStructPtr->minInputArray,
                 netPtr->netStructPtr->maxInputArray);

    ScaleUpFact (netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1],
                 netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],
                 netPtr->netStructPtr->minOutputArray,
                 netPtr->netStructPtr->maxOutputArray);

    ScaleUpFact (netPtr->netStructPtr->expectedOutputArray,
                 netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],
                 netPtr->netStructPtr->minOutputArray,
                 netPtr->netStructPtr->maxOutputArray);

    if (!netPtr->fileStructPtr->categoricalOutput) {
      int nOut = netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1];
      int oi;
      for (oi = 0; oi < nOut; oi++) {
        float diff = netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][oi]
                   - netPtr->netStructPtr->expectedOutputArray[oi];
        sumAbsErrorOrig += fabsf(diff);
        sumSqErrorOrig  += diff * diff;
      }
    }

    if (!factCorrect)
      {
        failedFacts++;
        WriteFailedFactsFileLine(netPtr,i, sumSquaredErrorThisFact);
      }

    WriteOutputFileLine(netPtr,MODE_TESTING);

  }

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));
  netPtr->netStructPtr->meanSumSquaredError = (float)sumSquaredError/(float)netPtr->fileStructPtr->numInputRows;
  netPtr->netStructPtr->percentFactsCorrect = 100.0f * (float)(netPtr->fileStructPtr->numInputRows-failedFacts) /
    (float)(netPtr->fileStructPtr->numInputRows);
  {
    int n = netPtr->fileStructPtr->numInputRows;
    netPtr->netStructPtr->mae  = (n > 0) ? sumAbsErrorOrig / (float)n : 0.0f;
    netPtr->netStructPtr->rmse = (n > 0) ? sqrtf(sumSqErrorOrig  / (float)n) : 0.0f;
  }
  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  WriteStatsFileLine(netPtr,
                     netPtr->netStructPtr->numberOfPasses, (failedFacts>0)?0:1,
                     netPtr->netStructPtr->meanSumSquaredError,
                     netPtr->netStructPtr->percentFactsCorrect);

  SendStatsPacket(netPtr);
  SendNetFinishedPacket(netPtr);

  pthread_detach(netPtr->thread);
  RemoveNetwork (netPtr);

  EXIT (func);
  return NULL;
}
