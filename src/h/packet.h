#ifndef PACKET_H
#define PACKET_H

#include "common.h"

#define COEF_PACKET 100 //inbound - get current system load
#define RUN_PACKET 101 //inbound - run a network
#define TRAIN_PACKET 102 //inbound - train a network, start with existing mtxfile if exists
#define RETRAIN_PACKET 103 //inbound - train a network, delete any exisitng mtxfile
#define TEST_PACKET 104 //inbound - test a network
#define STATS_PACKET 105 //outbound - sent async periodically while network running, training, testing
#define CANCEL_PACKET 106 //inbound - cancel a run, train, test network
#define ERROR_PACKET 107 //outbound - sent async when error discovered trying to run, train, test
#define NETFINISHED_PACKET 108 //outbound - sent after network is finished run, train, test
#define NUDGE_PACKET 109 //inbound - sent in to nudge the weights to try to escape local min
#define SHUTDOWN_PACKET 110 //inbound - shutsdown the entire process
#define REMOVENETWORK_PACKET 111 //inboud - remove the data structs and files for the specified network



/* packet struct */

typedef struct 
{
  int netID;
  int packetType;
  int ccode;
  char *status;
  char* src;
  char* dest;
  int neurons[MAX_LAYERS];
  int layers;	
  float learningRate;
  float momentum;
  float trainingTolerance;
  float trainingSuccessPercentage;
  float testingTolerance;
  float nudgeAmount;
  int   callbackPort;           /* port the client listens on for async callbacks (default: NNCD_TCP_PORT) */
  float weightDecay;            /* L2 regularisation coefficient (0 = disabled) */
  float dropoutRate;            /* fraction of hidden neurons zeroed per forward pass (0 = disabled) */
  int   useAdam;                /* 1 = Adam optimiser; 0 = SGD + momentum (default) */
  float adamBeta1;              /* Adam first-moment decay  (default 0.9)    */
  float adamBeta2;              /* Adam second-moment decay (default 0.999)  */
  float adamEpsilon;            /* Adam numerical stability (default 1e-8)   */
  char* mtxfile;
  char* infile;
  char* outfile;
  char *stsfile;
  char *failedfactsfile;
  float loadAvg;
  long memFree;
  long swapFree;
  int   cpuCount;          /* logical CPU cores (for load normalisation) */
  int   activeNetworks;    /* networks currently being trained/run/tested */
  int   gpuCount;          /* CUDA-capable GPUs on this server */
  long  gpuVramFreeMB;     /* free VRAM on the least-loaded GPU (MB) */
  int   gpuUtilPct;        /* utilisation % of the least-loaded GPU (0-100) */
  int   gpuComputeSM;      /* SM compute capability × 10, e.g. 89 for sm_89 */
  int inputFileHasHeader;
  int groundTruthFirst;    /* 1 = label/output column(s) are first; 0 = last (default) */
  int categoricalOutput;   /* 1 = single integer label → one-hot; 0 = raw multi-column output */
  int outputLabelFirst;    /* 1 = write predicted label as first output column; 0 = last (default) */
  int activationFunction;   /* ACTIVATION_SIGMOID / TANH / RELU / LEAKY_RELU */
  int numInputRows;
  float meanSumSquaredError;
  float percentFactsCorrect;
  float mae;   /* mean absolute error in original output units */
  float rmse;  /* root mean squared error in original output units */
  int numberOfPasses;
  char *timestamp;          /* wall-clock time when this packet was sent */
  long trainingDuration;    /* seconds elapsed since training started */
  long trainingTimeout;     /* server-side hard stop (seconds); 0 = unlimited */
}PacketStruct;

#endif /* PACKET_H */
