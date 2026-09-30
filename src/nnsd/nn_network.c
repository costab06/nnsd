/***************************************************************************
 *                      NNSD Network Lifecycle                              *
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "common.h"
#include "log_utils.h"
#include "nn_globals.h"
#include "nn_network.h"
#include "nn_fileio.h"
#include "packet.h"
#ifdef USE_CUDA
#include "nn_cuda.h"
#endif


/*********************** GetNetwork **************************************/
ThreadStruct *GetNetwork(int netID) {

  ThreadStruct *temp;

  pthread_mutex_lock (&threadListMtx);
  temp = threadInfoList;

  while (temp != NULL) {
    if (temp->netID != netID) {
      temp = temp->flink;
    } else {
      break;
    }
  }
  pthread_mutex_unlock (&threadListMtx);

  return temp;
}


/*************************** RemoveNetwork ******************************/
void
RemoveNetwork (ThreadStruct *netPtr)
{
  ThreadStruct *temp;
  char func[] = "RemoveNetwork";
  ENTER (func);

  DebugThreadList ("Before Remove");

  if (netPtr != NULL) {
    pthread_mutex_lock (&threadListMtx);

    if (netPtr == threadInfoList) {
      threadInfoList = netPtr->flink;
      netPtr->flink = NULL;
      DeallocateNetwork(netPtr);
    }
    else {
      temp = threadInfoList;
      while (temp != NULL) {
        if (temp->flink == netPtr) {
          temp->flink = netPtr->flink;
          netPtr->flink = NULL;
          DeallocateNetwork(netPtr);
          break;
        }
        temp = temp->flink;
      }
      if (temp == NULL) {
        LOG_ERROR("Network not found in list\n");
      }
    }

    pthread_mutex_unlock (&threadListMtx);

    DebugThreadList ("After Remove");
  }
  EXIT(func);
}


/*************************** DebugThreadList ******************************/
void
DebugThreadList (char *str)
{
  ThreadStruct *temp;
  char msg_buff[EIGHT_KB];
  char func[] = "DebugThreadList";
  ENTER (func);

  pthread_mutex_lock (&threadListMtx);

  temp = threadInfoList;

  if (temp == NULL) {
    sprintf(msg_buff,"%s: List is empty, threadInfoList is NULL\n",str);
    LOG_DEBUG_LEVEL1(msg_buff);
  }
  else {
    sprintf(msg_buff,"%s: List NOT empty, threadInfoList is %p\n",str,(void*)threadInfoList);
    LOG_DEBUG_LEVEL1(msg_buff);
    sprintf(msg_buff,"%s: threadInfoList->flink is %p\n",str,(void*)threadInfoList->flink);
    LOG_DEBUG_LEVEL1(msg_buff);

    int i=0;
    while (temp != NULL) {
      i++;
      if (temp->flink == NULL) {
        sprintf(msg_buff,"%s: Found network %d -> netID = %d, last in list - flink is NULL\n",str,i,temp->netID);
      } else {
        sprintf(msg_buff,"%s: Found network %d -> netID = %d\n",str,i,temp->netID);
      }
      LOG_DEBUG_LEVEL1(msg_buff);
      temp = temp->flink;
    }
  }

  pthread_mutex_unlock (&threadListMtx);

  EXIT(func);
}


/************************** ShutdownNetworks ******************************/
void
ShutdownNetworks ()
{
  char msg_buff[EIGHT_KB];
  char *func = "ShutdownNetworks";
  ENTER(func);

  pthread_mutex_lock (&threadListMtx);
  ThreadStruct* temp = threadInfoList;

  while (temp != NULL)
    {
      pthread_mutex_lock (&(temp->threadInfoHoldMtx));
      sprintf(msg_buff,"Setting statusFlag for netID %d to STATUS_SHUTDOWN\n",temp->netID);
      temp->statusFlag = STATUS_SHUTDOWN;
      pthread_mutex_unlock (&(temp->threadInfoHoldMtx));
      temp = temp->flink;
    }
  pthread_mutex_unlock (&threadListMtx);

  do
    {
      sleep(1);
    } while (threadInfoList != NULL);

  LOG_INFO("All networks/threads completed\n");
  EXIT(func);
}


/************************** AllocateWeightsAndVectors ***********************/
int AllocateWeightsAndVectors(ThreadStruct *netPtr) {

  char *func = "AllocateWeightsAndVectors";
  ENTER(func);

  int i; for (i = 0; i < netPtr->netStructPtr->layers; i++)
    {
      netPtr->netStructPtr->vectors[i] = calloc(netPtr->netStructPtr->neurons[i]+1,sizeof (float));
      netPtr->netStructPtr->deltas[i] = calloc(netPtr->netStructPtr->neurons[i]+1,sizeof (float));
    }

  for (i = 0; i < netPtr->netStructPtr->layers - 1; i++)
    {
      netPtr->netStructPtr->weights[i] = calloc(netPtr->netStructPtr->neurons[i]+1,sizeof (float *));
      netPtr->netStructPtr->previousWeightChange[i] = calloc(netPtr->netStructPtr->neurons[i]+1,sizeof (float *));
      if (netPtr->netStructPtr->useAdam) {
        netPtr->netStructPtr->adamM[i] = calloc(netPtr->netStructPtr->neurons[i]+1, sizeof(float *));
        netPtr->netStructPtr->adamV[i] = calloc(netPtr->netStructPtr->neurons[i]+1, sizeof(float *));
      }

      int j; for (j = 0; j < netPtr->netStructPtr->neurons[i]+1; j++) {
        netPtr->netStructPtr->weights[i][j] = calloc(netPtr->netStructPtr->neurons[i + 1],sizeof (float));
        netPtr->netStructPtr->previousWeightChange[i][j] = calloc(netPtr->netStructPtr->neurons[i + 1],sizeof (float));
        if (netPtr->netStructPtr->useAdam) {
          netPtr->netStructPtr->adamM[i][j] = calloc(netPtr->netStructPtr->neurons[i + 1], sizeof(float));
          netPtr->netStructPtr->adamV[i][j] = calloc(netPtr->netStructPtr->neurons[i + 1], sizeof(float));
        }
      }
    }

  netPtr->netStructPtr->minInputArray = calloc(netPtr->netStructPtr->neurons[0],sizeof (float));
  netPtr->netStructPtr->maxInputArray =  calloc(netPtr->netStructPtr->neurons[0],sizeof (float));

  netPtr->netStructPtr->minOutputArray =
    calloc(netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],sizeof (float));
  netPtr->netStructPtr->maxOutputArray =
    calloc(netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],sizeof (float));

  netPtr->netStructPtr->expectedOutputArray =
    calloc(netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1],sizeof (float));

  EXIT(func);
  return SUCCESS;
}


/************************** DeallocateNetwork *****************************/
int DeallocateNetwork(ThreadStruct *netPtr) {

  char *func = "DeallocateNetwork";
  ENTER(func);

  if (netPtr->src != NULL) {
    free(netPtr->src);
  }

  pthread_mutex_destroy (&(netPtr->threadInfoHoldMtx));

  if (netPtr->fileStructPtr != NULL) {

    if (netPtr->fileStructPtr->mtxfile != NULL)
      free(netPtr->fileStructPtr->mtxfile);

    if (netPtr->fileStructPtr->infile != NULL)
      free(netPtr->fileStructPtr->infile);

    if (netPtr->fileStructPtr->outfile != NULL)
      free(netPtr->fileStructPtr->outfile);

    if (netPtr->fileStructPtr->stsfile != NULL)
      free(netPtr->fileStructPtr->stsfile);

    if (netPtr->fileStructPtr->failedfactsfile != NULL)
      free(netPtr->fileStructPtr->failedfactsfile);

    if (netPtr->fileStructPtr->inputFileHasHeader == TRUE) {
      int i; for(i=0;i<netPtr->fileStructPtr->headerFields->numFields;i++) {
        free(netPtr->fileStructPtr->headerFields->headerData[i]);
      }
      free (netPtr->fileStructPtr->headerFields->headerData);
      free(netPtr->fileStructPtr->headerFields);
    }

    int i; for (i=0;i< netPtr->fileStructPtr->numInputRows;i++) {
      free(netPtr->fileStructPtr->inputRows[i].rowData);
    }

    free(netPtr->fileStructPtr->inputRows);
    free (netPtr->fileStructPtr);
  }

  if (netPtr->netStructPtr != NULL) {

#ifdef USE_CUDA
    if (netPtr->netStructPtr->cudaCtx) {
      CudaFree(netPtr->netStructPtr->cudaCtx, netPtr->netStructPtr->layers);
      netPtr->netStructPtr->cudaCtx = NULL;
    }
#endif

    if (netPtr->netStructPtr->minInputArray != NULL)
      free(netPtr->netStructPtr->minInputArray);

    if (netPtr->netStructPtr->maxInputArray != NULL)
      free(netPtr->netStructPtr->maxInputArray);

    if (netPtr->netStructPtr->minOutputArray != NULL)
      free(netPtr->netStructPtr->minOutputArray);

    if (netPtr->netStructPtr->maxOutputArray != NULL)
      free(netPtr->netStructPtr->maxOutputArray);

    if (netPtr->netStructPtr->expectedOutputArray != NULL)
      free(netPtr->netStructPtr->expectedOutputArray);

    int i; for (i = 0; i < netPtr->netStructPtr->layers; i++)
      {
        if (netPtr->netStructPtr->vectors[i] != NULL) {
          free(netPtr->netStructPtr->vectors[i]);
          free(netPtr->netStructPtr->deltas[i]);
        }
      }

    for (i = 0; i < netPtr->netStructPtr->layers - 1; i++)
      {
        int j; for (j=0;j<netPtr->netStructPtr->neurons[i]+1;j++) {
          if (netPtr->netStructPtr->weights[i][j] != NULL) {
            free(netPtr->netStructPtr->weights[i][j]);
            free(netPtr->netStructPtr->previousWeightChange[i][j]);
          }
          if (netPtr->netStructPtr->useAdam) {
            if (netPtr->netStructPtr->adamM[i] && netPtr->netStructPtr->adamM[i][j])
              free(netPtr->netStructPtr->adamM[i][j]);
            if (netPtr->netStructPtr->adamV[i] && netPtr->netStructPtr->adamV[i][j])
              free(netPtr->netStructPtr->adamV[i][j]);
          }
        }
        free(netPtr->netStructPtr->weights[i]);
        free(netPtr->netStructPtr->previousWeightChange[i]);
        if (netPtr->netStructPtr->useAdam) {
          if (netPtr->netStructPtr->adamM[i]) free(netPtr->netStructPtr->adamM[i]);
          if (netPtr->netStructPtr->adamV[i]) free(netPtr->netStructPtr->adamV[i]);
        }
      }
  }
  free (netPtr->netStructPtr);
  free (netPtr);

  EXIT(func);
  return SUCCESS;
}


/*************************** InitializeNetworkFromPacket *****************************/
int InitializeNetworkFromPacket (PacketStruct * packet, ThreadStruct **netPtrAddr) {

  int ccode = SUCCESS;
  char msg_buff[EIGHT_KB];
  char *func = "InitializeNetworkFromPacket";
  ENTER (func);
  assert(packet != NULL);

  ThreadStruct *tempNetPtr;

  *netPtrAddr  = (ThreadStruct *) calloc (1, sizeof (ThreadStruct));
  tempNetPtr = *netPtrAddr;
  if (tempNetPtr == NULL)
    return MALLOCERROR;
  tempNetPtr->netStructPtr = (NetStruct *) calloc(1,sizeof (NetStruct));
  if (tempNetPtr->netStructPtr == NULL)
    return MALLOCERROR;
  tempNetPtr->fileStructPtr = (FileStruct *) calloc(1,sizeof(FileStruct));
  if (tempNetPtr->fileStructPtr == NULL)
    return MALLOCERROR;

  tempNetPtr->netID = packet->netID;
  tempNetPtr->networkType = packet->packetType;
  tempNetPtr->src = strdup(packet->src);
  tempNetPtr->callbackPort = (packet->callbackPort > 0) ? packet->callbackPort : NNCD_TCP_PORT;
  tempNetPtr->statusFlag = STATUS_RUN;
  tempNetPtr->nudgeAmount = packet->nudgeAmount;
  tempNetPtr->flink = NULL;

  pthread_mutex_init
    (&(tempNetPtr->threadInfoHoldMtx),
     (pthread_mutexattr_t *) NULL);

  memcpy(tempNetPtr->netStructPtr->neurons,packet->neurons,sizeof(packet->neurons));
  tempNetPtr->netStructPtr->layers = packet->layers;
  tempNetPtr->netStructPtr->learningRate = packet->learningRate;
  tempNetPtr->netStructPtr->momentum =  packet->momentum;
  tempNetPtr->netStructPtr->trainingTolerance = packet->trainingTolerance;
  tempNetPtr->netStructPtr->trainingSuccessPercentage = packet->trainingSuccessPercentage;
  tempNetPtr->netStructPtr->testingTolerance = packet->testingTolerance;
  tempNetPtr->netStructPtr->weightDecay  = packet->weightDecay;
  tempNetPtr->netStructPtr->dropoutRate  = packet->dropoutRate;
  tempNetPtr->netStructPtr->useAdam      = packet->useAdam;
  tempNetPtr->netStructPtr->adamBeta1    = (packet->adamBeta1  > 0.0f) ? packet->adamBeta1  : 0.9f;
  tempNetPtr->netStructPtr->adamBeta2    = (packet->adamBeta2  > 0.0f) ? packet->adamBeta2  : 0.999f;
  tempNetPtr->netStructPtr->adamEpsilon  = (packet->adamEpsilon > 0.0f) ? packet->adamEpsilon : 1e-8f;
  tempNetPtr->netStructPtr->adamT        = 0L;
  tempNetPtr->netStructPtr->trainingTimeout    = packet->trainingTimeout;
  tempNetPtr->netStructPtr->activationFunction = packet->activationFunction;

  AllocateWeightsAndVectors(tempNetPtr);

  tempNetPtr->fileStructPtr->mtxfile = strdup(packet->mtxfile);
  tempNetPtr->fileStructPtr->infile = strdup(packet->infile);
  tempNetPtr->fileStructPtr->outfile = strdup(packet->outfile);
  tempNetPtr->fileStructPtr->stsfile = strdup(packet->stsfile);
  tempNetPtr->fileStructPtr->failedfactsfile = strdup(packet->failedfactsfile);
  tempNetPtr->fileStructPtr->inputFileHasHeader = packet->inputFileHasHeader;
  tempNetPtr->fileStructPtr->groundTruthFirst   = packet->groundTruthFirst;
  tempNetPtr->fileStructPtr->categoricalOutput  = packet->categoricalOutput;
  tempNetPtr->fileStructPtr->outputLabelFirst   = packet->outputLabelFirst;

  sprintf(msg_buff,"Initializing weight data to low random values\n");

  int i; for (i = 0; i < tempNetPtr->netStructPtr->layers - 1; i++)
    {
      int j; for (j = 0; j < tempNetPtr->netStructPtr->neurons[i]+1; j++)
        {
          int k; for (k = 0; k < tempNetPtr->netStructPtr->neurons[i + 1]; k++)
            {
              float randWeight = ((float)(rand() % 201 - 100)) / 500.0f;
              tempNetPtr->netStructPtr->weights[i][j][k] = randWeight;
            }
        }
    }

  LoadAndValidateInputFile(tempNetPtr);

  int gtfLabelCols    = tempNetPtr->fileStructPtr->categoricalOutput
                          ? 1
                          : tempNetPtr->netStructPtr->neurons[tempNetPtr->netStructPtr->layers - 1];
  int gtfInputOffset  = tempNetPtr->fileStructPtr->groundTruthFirst ? gtfLabelCols : 0;
  int gtfOutputOffset = tempNetPtr->fileStructPtr->groundTruthFirst
                          ? 0
                          : tempNetPtr->netStructPtr->neurons[0];

  for (i = 0; i < tempNetPtr->netStructPtr->neurons[0]; i++)
    {
      tempNetPtr->netStructPtr->minInputArray[i] = tempNetPtr->fileStructPtr->inputRows[0].rowData[gtfInputOffset + i];
      tempNetPtr->netStructPtr->maxInputArray[i] = tempNetPtr->fileStructPtr->inputRows[0].rowData[gtfInputOffset + i];
    }

  for (i = 0; i < tempNetPtr->netStructPtr->neurons[tempNetPtr->netStructPtr->layers - 1]; i++)
    {
      if (tempNetPtr->fileStructPtr->categoricalOutput) {
        tempNetPtr->netStructPtr->minOutputArray[i] = 0.0f;
        tempNetPtr->netStructPtr->maxOutputArray[i] = 1.0f;
      } else {
        tempNetPtr->netStructPtr->minOutputArray[i] =
          tempNetPtr->fileStructPtr->inputRows[0].rowData[gtfOutputOffset + i];
        tempNetPtr->netStructPtr->maxOutputArray[i] =
          tempNetPtr->fileStructPtr->inputRows[0].rowData[gtfOutputOffset + i];
      }
    }

  sprintf(msg_buff,"Looping through rest of facts: %d",(tempNetPtr->fileStructPtr->numInputRows - 1));

  for (i = 1; i < tempNetPtr->fileStructPtr->numInputRows; i++)
    {
      int j; for (j = 0; j < tempNetPtr->netStructPtr->neurons[0]; j++)
        {
          if (tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfInputOffset + j] < tempNetPtr->netStructPtr->minInputArray[j]) {
            sprintf(msg_buff,"Updating minInputArray[%d] = %f",j,tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfInputOffset + j]);
            LOG_DEBUG_LEVEL2(msg_buff);
            tempNetPtr->netStructPtr->minInputArray[j] = tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfInputOffset + j];
          }
          if (tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfInputOffset + j] > tempNetPtr->netStructPtr->maxInputArray[j]) {
            sprintf(msg_buff,"Updating maxInputArray[%d] = %f",j,tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfInputOffset + j]);
            LOG_DEBUG_LEVEL2(msg_buff);
            tempNetPtr->netStructPtr->maxInputArray[j] = tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfInputOffset + j];
          }
        }

      if (!tempNetPtr->fileStructPtr->categoricalOutput) {
        for (j = 0; j < tempNetPtr->netStructPtr->neurons[tempNetPtr->netStructPtr->layers - 1]; j++)
          {
            if (tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfOutputOffset + j] <
                tempNetPtr->netStructPtr->minOutputArray[j]) {
              tempNetPtr->netStructPtr->minOutputArray[j] =
                tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfOutputOffset + j];
            }
            if (tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfOutputOffset + j] >
                tempNetPtr->netStructPtr->maxOutputArray[j]) {
              tempNetPtr->netStructPtr->maxOutputArray[j] =
                tempNetPtr->fileStructPtr->inputRows[i].rowData[gtfOutputOffset + j];
            }
          }
      }
    }

  for (i = 0; i < tempNetPtr->netStructPtr->neurons[0]; i++) {
    float range = (tempNetPtr->netStructPtr->maxInputArray[i] - tempNetPtr->netStructPtr->minInputArray[i]);
    float extendedRange = range/0.8;
    float addedRange = extendedRange - range;
    tempNetPtr->netStructPtr->minInputArray[i] =
      tempNetPtr->netStructPtr->minInputArray[i] - (addedRange / 2.0);
    tempNetPtr->netStructPtr->maxInputArray[i] =
      tempNetPtr->netStructPtr->maxInputArray[i] + (addedRange / 2.0);
  }

  for (i = 0; i < tempNetPtr->netStructPtr->neurons[tempNetPtr->netStructPtr->layers - 1]; i++) {
    float range = (tempNetPtr->netStructPtr->maxOutputArray[i] - tempNetPtr->netStructPtr->minOutputArray[i]);
    float extendedRange = range/0.8;
    float addedRange = extendedRange - range;
    tempNetPtr->netStructPtr->minOutputArray[i] =
      tempNetPtr->netStructPtr->minOutputArray[i] - (addedRange / 2.0);
    tempNetPtr->netStructPtr->maxOutputArray[i] =
      tempNetPtr->netStructPtr->maxOutputArray[i] + (addedRange / 2.0);
  }

  WriteMatrixFile(tempNetPtr);

#ifdef USE_CUDA
  if (tempNetPtr->networkType == TRAIN_PACKET ||
      tempNetPtr->networkType == RETRAIN_PACKET) {
    tempNetPtr->netStructPtr->cudaCtx = CudaInit(tempNetPtr);
  }
#endif

  DebugThreadList("Before Add");

  pthread_mutex_lock (&threadListMtx);
  tempNetPtr->flink = threadInfoList;
  threadInfoList = tempNetPtr;
  pthread_mutex_unlock (&threadListMtx);

  DebugThreadList("After Add");

  EXIT(func);
  return ccode;
}


/*************************** InitializeNetworkFromMatrixFile *****************************/
int InitializeNetworkFromMatrixFile (PacketStruct * packet, ThreadStruct **netPtrAddr) {

  int ccode = SUCCESS;
  char *mode="r";
  FILE *mfp;
  char msg_buff[EIGHT_KB];

  char *func = "InitializeNetworkFromMatrixFile";
  ENTER (func);
  assert(packet != NULL);

  (*netPtrAddr)  = (ThreadStruct *) calloc (1, sizeof (ThreadStruct));
  if (*netPtrAddr == NULL)
    return MALLOCERROR;

  (*netPtrAddr)->netStructPtr = (NetStruct *) calloc(1,sizeof (NetStruct));
  if ((*netPtrAddr)->netStructPtr == NULL)
    return MALLOCERROR;

  (*netPtrAddr)->fileStructPtr = (FileStruct *) calloc(1,sizeof(FileStruct));
  if ((*netPtrAddr)->fileStructPtr == NULL)
    return MALLOCERROR;

  (*netPtrAddr)->netID = packet->netID;
  (*netPtrAddr)->networkType = packet->packetType;
  (*netPtrAddr)->src = strdup(packet->src);
  (*netPtrAddr)->callbackPort = (packet->callbackPort > 0) ? packet->callbackPort : NNCD_TCP_PORT;
  (*netPtrAddr)->statusFlag = STATUS_RUN;
  (*netPtrAddr)->netStructPtr->trainingTimeout = packet->trainingTimeout;
  (*netPtrAddr)->nudgeAmount = packet->nudgeAmount;
  (*netPtrAddr)->flink = NULL;

  pthread_mutex_init
    (&((*netPtrAddr)->threadInfoHoldMtx),
     (pthread_mutexattr_t *) NULL);

  (*netPtrAddr)->fileStructPtr->mtxfile = strdup(packet->mtxfile);
  (*netPtrAddr)->fileStructPtr->infile = strdup(packet->infile);
  (*netPtrAddr)->fileStructPtr->outfile = strdup(packet->outfile);
  (*netPtrAddr)->fileStructPtr->stsfile = strdup(packet->stsfile);
  (*netPtrAddr)->fileStructPtr->failedfactsfile = strdup(packet->failedfactsfile);
  (*netPtrAddr)->fileStructPtr->inputFileHasHeader = packet->inputFileHasHeader;
  (*netPtrAddr)->fileStructPtr->groundTruthFirst   = packet->groundTruthFirst;
  (*netPtrAddr)->fileStructPtr->categoricalOutput  = packet->categoricalOutput;
  (*netPtrAddr)->fileStructPtr->outputLabelFirst   = packet->outputLabelFirst;

  mfp=fopen((*netPtrAddr)->fileStructPtr->mtxfile, mode);
  if ( mfp == NULL) {
    sprintf (msg_buff, "file open error for %s, mode %s - errno: %d\n",
             (*netPtrAddr)->fileStructPtr->mtxfile, mode, errno);
    LOG_ERROR (msg_buff);
    ccode = FOPENERROR;
    return ccode;
  }

  if (fread (&((*netPtrAddr)->netStructPtr->learningRate), sizeof(float), 1, mfp) != 1) {
    LOG_ERROR ("Cannot read learningRate\n");
    ccode = FREADERROR; return ccode;
  }
  if (fread (&((*netPtrAddr)->netStructPtr->momentum), sizeof(float), 1, mfp) != 1) {
    LOG_ERROR ("Cannot read Momentum\n");
    ccode = FREADERROR; return ccode;
  }
  if (fread (&((*netPtrAddr)->netStructPtr->trainingTolerance), sizeof(float), 1, mfp) != 1) {
    LOG_ERROR ("Cannot read trainingTolerance\n");
    ccode = FREADERROR; return ccode;
  }
  if (fread (&((*netPtrAddr)->netStructPtr->trainingSuccessPercentage), sizeof(float), 1, mfp) != 1) {
    LOG_ERROR ("Cannot read trainingSuccessPercentage\n");
    ccode = FREADERROR; return ccode;
  }
  if (fread (&((*netPtrAddr)->netStructPtr->testingTolerance), sizeof(float), 1, mfp) != 1) {
    LOG_ERROR ("Cannot read testingTolerance\n");
    ccode = FREADERROR; return ccode;
  }
  if (fread (&((*netPtrAddr)->netStructPtr->activationFunction), sizeof(int), 1, mfp) != 1) {
    (*netPtrAddr)->netStructPtr->activationFunction = ACTIVATION_SIGMOID;
  }
  if (fread (&((*netPtrAddr)->netStructPtr->layers), sizeof(int), 1, mfp) != 1) {
    LOG_ERROR ("cannot read layers\n");
    ccode = FREADERROR; return ccode;
  }
  if (fread (&((*netPtrAddr)->netStructPtr->neurons), sizeof(int)*MAX_LAYERS, 1, mfp) != 1) {
    LOG_ERROR ("cannot read neurons\n");
    ccode = FREADERROR; return ccode;
  }

  sprintf (msg_buff,
           "Read in Matrix File values:\n learningRate = %f\n momentum = %f\n trainingTolerance = %f\n trainingSuccessPercentage = %f\n testingTolerance = %f\n layers = %d\n neurons: %d %d %d %d %d %d %d %d\n",
           (*netPtrAddr)->netStructPtr->learningRate,
           (*netPtrAddr)->netStructPtr->momentum,
           (*netPtrAddr)->netStructPtr->trainingTolerance,
           (*netPtrAddr)->netStructPtr->trainingSuccessPercentage,
           (*netPtrAddr)->netStructPtr->testingTolerance,
           (*netPtrAddr)->netStructPtr->layers,
           (*netPtrAddr)->netStructPtr->neurons[0],
           (*netPtrAddr)->netStructPtr->neurons[1],
           (*netPtrAddr)->netStructPtr->neurons[2],
           (*netPtrAddr)->netStructPtr->neurons[3],
           (*netPtrAddr)->netStructPtr->neurons[4],
           (*netPtrAddr)->netStructPtr->neurons[5],
           (*netPtrAddr)->netStructPtr->neurons[6],
           (*netPtrAddr)->netStructPtr->neurons[7]);

  AllocateWeightsAndVectors(*netPtrAddr);

  int i; for (i = 0; i < (*netPtrAddr)->netStructPtr->layers - 1; i++) {
    int j; for (j = 0; j < ((*netPtrAddr)->netStructPtr->neurons[i]+1); j++) {
      if (fread ((*netPtrAddr)->netStructPtr->weights[i][j],
                 sizeof (float),
                 (size_t)((*netPtrAddr)->netStructPtr->neurons[i+1]),
                 mfp) != (size_t)((*netPtrAddr)->netStructPtr->neurons[i+1])) {
        sprintf (msg_buff, "Error reading weights from matrix file: %s with mode %s: %d\n",
                 (*netPtrAddr)->fileStructPtr->mtxfile, mode, errno);
        LOG_ERROR (msg_buff);
      }
    }
  }

  for (i = 0; i < (*netPtrAddr)->netStructPtr->neurons[0]; i++) {
    if (fread(&((*netPtrAddr)->netStructPtr->minInputArray[i]), sizeof (float), 1, mfp) != 1)
      return FREADERROR;
  }
  for (i = 0;i < (*netPtrAddr)->netStructPtr->neurons[(*netPtrAddr)->netStructPtr->layers - 1]; i++) {
    if (fread (&((*netPtrAddr)->netStructPtr->minOutputArray[i]), sizeof (float), 1, mfp) != 1)
      ccode = FREADERROR;
  }
  for (i = 0; i < (*netPtrAddr)->netStructPtr->neurons[0]; i++) {
    if (fread(&((*netPtrAddr)->netStructPtr->maxInputArray[i]), sizeof (float), 1, mfp) != 1)
      ccode = FREADERROR;
  }
  for (i = 0;i < (*netPtrAddr)->netStructPtr->neurons[(*netPtrAddr)->netStructPtr->layers - 1]; i++) {
    if (fread (&((*netPtrAddr)->netStructPtr->maxOutputArray[i]), sizeof (float), 1, mfp) != 1)
      ccode = FREADERROR;
  }

  fclose(mfp);

  /* Fields not stored in .mtx — take from packet so retraining respects them. */
  (*netPtrAddr)->netStructPtr->dropoutRate  = packet->dropoutRate;
  (*netPtrAddr)->netStructPtr->weightDecay  = packet->weightDecay;
  (*netPtrAddr)->netStructPtr->useAdam      = packet->useAdam;
  (*netPtrAddr)->netStructPtr->adamBeta1    = (packet->adamBeta1  > 0.0f) ? packet->adamBeta1  : 0.9f;
  (*netPtrAddr)->netStructPtr->adamBeta2    = (packet->adamBeta2  > 0.0f) ? packet->adamBeta2  : 0.999f;
  (*netPtrAddr)->netStructPtr->adamEpsilon  = (packet->adamEpsilon > 0.0f) ? packet->adamEpsilon : 1e-8f;
  (*netPtrAddr)->netStructPtr->adamT        = 0L;
  /* Allow caller to override trainingSuccessPercentage (e.g. tox21_demo sends 75). */
  if (packet->trainingSuccessPercentage > 0.0f)
    (*netPtrAddr)->netStructPtr->trainingSuccessPercentage = packet->trainingSuccessPercentage;

  LoadAndValidateInputFile((*netPtrAddr));

#ifdef USE_CUDA
  if ((*netPtrAddr)->networkType == TRAIN_PACKET ||
      (*netPtrAddr)->networkType == RETRAIN_PACKET) {
    (*netPtrAddr)->netStructPtr->cudaCtx = CudaInit(*netPtrAddr);
  }
#endif

  pthread_mutex_lock (&threadListMtx);
  (*netPtrAddr)->flink = threadInfoList;
  threadInfoList = (*netPtrAddr);
  pthread_mutex_unlock (&threadListMtx);

  EXIT(func);
  return ccode;
}
