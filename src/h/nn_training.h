#ifndef NN_TRAINING_H
#define NN_TRAINING_H

#include <semaphore.h>
#include "nnsd.h"
#include "common.h"

#define FOUR_SECONDS  4
#define SIXTY_SECONDS 60

typedef struct {
    ThreadStruct *netPtr;
    /* Dispatch (semaphore-based, no spinning) */
    sem_t         work_sem;
    sem_t         done_sem;
    pthread_t     thread;
    int           factIndex;        /* -1 = shutdown signal */
    /* Thread-local activation buffers */
    float        *vectors[MAX_LAYERS];
    float        *deltas[MAX_LAYERS];
    float        *expectedOutput;
    /* Per-worker gradient accumulator (mirrors weights layout) */
    float       **localGrad[MAX_LAYERS];
    /* Dropout masks for hidden layers (1 = kept, 0 = dropped) */
    float        *dropout_mask[MAX_LAYERS];
    unsigned int  rng_state;    /* per-worker PRNG seed for rand_r() */
    /* Results written back before posting done_sem */
    float         sumSquaredError;
    int           factCorrect;          /* tolerance-based: decides backprop */
    int           factPredictedCorrect; /* argmax-based: used for progress reporting */
} PoolWorker;

void *RunNetwork(void *arg);
void *TrainNetwork(void *arg);
void *TestNetwork(void *arg);

#endif /* NN_TRAINING_H */
