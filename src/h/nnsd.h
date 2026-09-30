#ifndef NNSD_H
#define NNSD_H

				/* Global Data Structures */

typedef struct {
  int headerIsDone;
  int numFields;
  char **headerData;
} HeaderStruct;



typedef struct {
  int terminationChar;
  int numFields;
  int firstConversionErrorField;
  float *rowData;
} RowStruct;


typedef struct {
  char* mtxfile;
  char* infile;
  char* outfile;
  char *stsfile;
  char *failedfactsfile;
  int inputFileHasHeader;
  int groundTruthFirst;    /* 1 = label/output column(s) are first; 0 = last (default) */
  int categoricalOutput;   /* 1 = single integer label → one-hot; 0 = raw multi-column output */
  int outputLabelFirst;    /* 1 = write predicted label as first output column; 0 = last (default) */
  HeaderStruct *headerFields;
  int numInputRows;
  RowStruct *inputRows; // row/field format
} FileStruct;


typedef struct {
  int neurons[MAX_LAYERS];
  int layers;	
  float learningRate;
  float momentum;
  float trainingTolerance;
  float trainingSuccessPercentage;
  float testingTolerance;
  float weightDecay;            /* L2 regularisation coefficient (0 = disabled) */
  float dropoutRate;            /* fraction of hidden neurons zeroed per forward pass (0 = disabled) */
  int   useAdam;                /* 1 = Adam optimiser; 0 = SGD + momentum (default) */
  float adamBeta1;              /* Adam first-moment decay  (default 0.9)    */
  float adamBeta2;              /* Adam second-moment decay (default 0.999)  */
  float adamEpsilon;            /* Adam numerical stability (default 1e-8)   */
  long  adamT;                  /* Adam step counter; incremented each batch */

  float meanSumSquaredError;
  float percentFactsCorrect;
  int numberOfPasses;
  float mae;   /* mean absolute error in original (de-scaled) output units */
  float rmse;  /* root mean squared error in original (de-scaled) output units */

  float *vectors[MAX_LAYERS];
  float **weights[MAX_LAYERS];

  float **previousWeightChange[MAX_LAYERS];
  float *deltas[MAX_LAYERS];

  float *minInputArray;
  float *maxInputArray;
  float *minOutputArray;
  float *maxOutputArray;

  float *expectedOutputArray;

  int activationFunction;     /* ACTIVATION_SIGMOID / TANH / RELU / LEAKY_RELU */

  time_t trainingStartTime;   /* set when training begins; used to compute duration */
  long   trainingTimeout;     /* server-side hard stop (seconds); 0 = unlimited */

  /* Mini-batch gradient accumulator — mirrors weights layout.
   * Allocated only during training; NULL otherwise. */
  float **gradAccum[MAX_LAYERS];  /* mirrors weights layout: gradAccum[layer][from][to] */

  /* Adam moment arrays — same shape as weights; allocated only when useAdam=1. */
  float **adamM[MAX_LAYERS];    /* first  moment (mean of gradients)          */
  float **adamV[MAX_LAYERS];    /* second moment (uncentred variance)         */

  /* CUDA GPU context — CudaContext* when USE_CUDA and a GPU is available,
   * NULL otherwise.  Opaque void* so nnsd.h stays free of CUDA headers. */
  void *cudaCtx;

} NetStruct;







typedef struct ThreadStructure
{
  int netID;
  int networkType;
  char *src;
  int callbackPort;             /* port on src to send async callbacks to */
  int statusFlag;			/* 0 - STATUS_RUN */
  float nudgeAmount;
  struct ThreadStructure *flink;
  pthread_mutex_t threadInfoHoldMtx;
  pthread_t thread;
  
  FileStruct *fileStructPtr;
  NetStruct *netStructPtr;

} ThreadStruct;

#endif /* NNSD_H */
