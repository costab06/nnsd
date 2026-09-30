#ifndef NN_FILEIO_H
#define NN_FILEIO_H

#include "nnsd.h"

int LoadAndValidateInputFile(ThreadStruct *netPtr);
int WriteMatrixFile(ThreadStruct *netPtr);
int BackupMatrixFile(ThreadStruct *netPtr);
int WriteStatsFileHeader(ThreadStruct *netPtr);
int WriteStatsFileLine(ThreadStruct *netPtr, int filePass, int passFail, float sumSquaredError, float percentCorrect);
int WriteOutputFileHeader(ThreadStruct *netPtr, int mode);
int WriteOutputFileLine(ThreadStruct *netPtr, int mode);
int WriteFailedFactsFileHeader(ThreadStruct *netPtr);
int WriteFailedFactsFileLine(ThreadStruct *netPtr, int factNumber, float sumSquaredErrorThisFact);

#endif /* NN_FILEIO_H */
