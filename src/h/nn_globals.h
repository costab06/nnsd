#ifndef NN_GLOBALS_H
#define NN_GLOBALS_H

#include <pthread.h>
#include "nnsd.h"

/* Global variables defined in nnsd.c */

extern ThreadStruct    *threadInfoList;
extern pthread_mutex_t  threadListMtx;

extern int   statusFlag;
extern int   runAsDaemon;
extern int   writeTrainingFailedFacts;
extern int   restPort;
extern char *mountPoint;
extern char *networkInterface;
extern char *ipAddress;

/* Mutexes defined in log_utils.c / tcp_utils.c */
extern pthread_mutex_t debugfileMtx;
extern pthread_mutex_t jSONParserMtx;

/* Debug flags defined in log_utils.c */
extern int debugPackets;
extern int debugMath;
extern int debugEnterExit;
extern int debugLevel1;
extern int debugLevel2;
extern int debugWeights;

/* Status flag accessors defined in nnsd.c */
int SetStatusFlag(int status);
int GetStatusFlag(void);
int SetNetStatusFlag(ThreadStruct *netPtr, int status);
int GetNetStatusFlag(ThreadStruct *netPtr);

#endif /* NN_GLOBALS_H */
