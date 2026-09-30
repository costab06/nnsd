/***************************************************************************
 *                          NNSD File I/O                                   *
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>

#include "common.h"
#include "log_utils.h"
#include "nn_globals.h"
#include "nn_fileio.h"
#include "csv.h"
#include "packet.h"


/* ── CSV helper functions ───────────────────────────────────────────────── */

static int is_space(unsigned char c) {
  if (c == CSV_SPACE || c == CSV_TAB) return 1;
  return 0;
}

static int is_term(unsigned char c) {
  if (c == CSV_CR || c == CSV_LF) return 1;
  return 0;
}


void EndOfFieldCallback (void *s, size_t len, void *data) {
  (void)len;
  char msg_buff[EIGHT_KB];

  ThreadStruct *netPtr = (ThreadStruct *) data;

  if (netPtr->fileStructPtr->inputFileHasHeader == TRUE && netPtr->fileStructPtr->headerFields->headerIsDone == FALSE) {

    if (netPtr->fileStructPtr->headerFields->headerData == NULL) {
      netPtr->fileStructPtr->headerFields->headerData = calloc(1, sizeof(char **));
      if (netPtr->fileStructPtr->headerFields->headerData == NULL) {
        sprintf(msg_buff,"Calloc of first element in headerFields is null");
        LOG_ERROR(msg_buff);
        return;
      }
    } else {
      void *tmp;
      tmp = realloc(netPtr->fileStructPtr->headerFields->headerData,sizeof(char **) * (netPtr->fileStructPtr->headerFields->numFields+1));
      if (tmp == NULL) {
        sprintf(msg_buff,"Realloc of element in headerFields is null");
        LOG_ERROR(msg_buff);
        return;
      } else {
        netPtr->fileStructPtr->headerFields->headerData = tmp;
      }
    }

    netPtr->fileStructPtr->headerFields->headerData[netPtr->fileStructPtr->headerFields->numFields] = strdup(s);
    netPtr->fileStructPtr->headerFields->numFields++;

  } else {

    if (netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].rowData == NULL) {
      netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].rowData = calloc(1,sizeof(float));
      if (netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].rowData == NULL) {
        sprintf(msg_buff,"Allocation of element in row is null");
        LOG_ERROR(msg_buff);
        return;
      }
    } else {
      void *tmp;
      tmp = realloc(netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].rowData,
                    sizeof(float)*(netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].numFields+1));
      if (tmp == NULL) {
        sprintf(msg_buff,"Reallocation of element in row is null");
        LOG_ERROR(msg_buff);
        return;
      } else {
        netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].rowData = tmp;
      }
    }

    char *endp = NULL;
    float val = strtof((const char *)s, &endp);

    netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].
      rowData[netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].numFields] = val;
    if (endp == (const char *)s) {
      if (netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].firstConversionErrorField < 0) {
        netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].firstConversionErrorField =
          netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].numFields;
      }
    }

    netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].numFields++;
  }
}


void EndOfRowCallback(int c, void *data) {
  char msg_buff[EIGHT_KB];

  ThreadStruct *netPtr = (ThreadStruct *) data;

  if (netPtr->fileStructPtr->inputFileHasHeader == TRUE && netPtr->fileStructPtr->headerFields->headerIsDone != TRUE) {
    netPtr->fileStructPtr->headerFields->headerIsDone = TRUE;
  } else {

    netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].terminationChar = c;

    void *tmp = realloc(netPtr->fileStructPtr->inputRows,
                        sizeof(RowStruct) * (netPtr->fileStructPtr->numInputRows+2));

    if (tmp == NULL) {
      sprintf(msg_buff,"Reallocation of element in row is null");
      LOG_ERROR(msg_buff);
      return;
    } else {
      netPtr->fileStructPtr->inputRows = tmp;
      netPtr->fileStructPtr->numInputRows++;

      netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].numFields=0;
      netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].firstConversionErrorField=-1;
      netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].rowData=NULL;
    }
  }
}


/*************************** LoadAndValidateInputFile *****************************/
int LoadAndValidateInputFile(ThreadStruct *netPtr) {

  struct csv_parser csvParser;
  unsigned char options = CSV_APPEND_NULL;

  FILE *fp;
  char buf[1024];
  char msg_buff[EIGHT_KB];
  size_t bytes_read;
  int ccode =0;

  char *func="LoadAndValidateInputFile";
  ENTER(func);

  if (csv_init(&csvParser, options) != 0) {
    sprintf(msg_buff, "Failed to initialize csv parser\n");
    LOG_ERROR(msg_buff);
    return FREADERROR;
  }

  csv_set_space_func(&csvParser, is_space);
  csv_set_term_func(&csvParser, is_term);

  fp = fopen(netPtr->fileStructPtr->infile, "rb");
  if (!fp) {
    sprintf(msg_buff, "Failed to open %s: %s\n", netPtr->fileStructPtr->infile, strerror(errno));
    LOG_ERROR(msg_buff);
    return FOPENERROR;
  }

  if (netPtr->fileStructPtr->inputFileHasHeader) {
    netPtr->fileStructPtr->headerFields = calloc(1, sizeof (HeaderStruct));
  }

  netPtr->fileStructPtr->inputRows = calloc(1, sizeof (RowStruct));
  netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].numFields=0;
  netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].firstConversionErrorField=-1;
  netPtr->fileStructPtr->inputRows[netPtr->fileStructPtr->numInputRows].rowData=NULL;

  while ((bytes_read=fread(buf, 1, 1024, fp)) > 0) {
    if (csv_parse(&csvParser, buf, bytes_read, EndOfFieldCallback, EndOfRowCallback, netPtr) != bytes_read) {
      sprintf(msg_buff, "Error while parsing file: %s\n", csv_strerror(csv_error(&csvParser)));
      LOG_ERROR(msg_buff);
      return FREADERROR;
    }
  }

  csv_fini(&csvParser, EndOfFieldCallback, EndOfRowCallback, netPtr);

  if (ferror(fp)) {
    sprintf(msg_buff, "Error while reading %s\n", netPtr->fileStructPtr->infile);
    LOG_ERROR(msg_buff);
    fclose(fp);
    return FREADERROR;
  }

  fclose(fp);
  csv_free(&csvParser);

  int columns = netPtr->netStructPtr->neurons[0];

  if ((netPtr->networkType == TRAIN_PACKET) ||
      (netPtr->networkType == RETRAIN_PACKET) ||
      (netPtr->networkType == TEST_PACKET))
    columns+=netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers -1];

  if (netPtr->fileStructPtr->inputFileHasHeader == TRUE && columns != netPtr->fileStructPtr->headerFields->numFields) {
    sprintf(msg_buff,"ERROR: file should have %d columns, but header has %d fields\n",
            columns, netPtr->fileStructPtr->headerFields->numFields);
    ccode = MALFORMEDINPUTFILEERROR;
    EXIT(func);
    return(ccode);
  }

  int i; for (i=0;i<netPtr->fileStructPtr->numInputRows;i++) {
    if (columns != netPtr->fileStructPtr->inputRows[i].numFields) {
      sprintf(msg_buff,"ERROR: row %d should have %d fields, but has %d fields\n",i, columns,
              netPtr->fileStructPtr->inputRows[i].numFields);
      ccode = MALFORMEDINPUTFILEERROR;
    }
  }

  EXIT(func);
  return ccode;
}


/*********************** BackupMatrixFile ***************************************/
int
BackupMatrixFile (ThreadStruct * netPtr)
{
  int ccode = 0;
  char backupFile[ONE_KB];
  FILE *src = NULL;
  FILE *dst = NULL;
  char buffer[EIGHT_KB];
  char msg_buff[EIGHT_KB];

  char func[] = "BackupMatrixFile";

  ENTER (func);

  strcpy(backupFile, netPtr->fileStructPtr->mtxfile);
  char *dot = strrchr(backupFile, '.');
  if (dot) *dot = '\0';
  strcat(backupFile, ".bak");

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));

  if (access (netPtr->fileStructPtr->mtxfile, R_OK))
    {
      sprintf(msg_buff,"access returned error for: %s\n",netPtr->fileStructPtr->mtxfile);
      LOG_ERROR (msg_buff);
      pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));
      return FILENOTEXISTERROR;
    }

  src = fopen (netPtr->fileStructPtr->mtxfile, "r");
  dst = fopen (backupFile, "w");

  if ((src != NULL) && (dst != NULL))
    {
      unsigned bytes;
      while ((bytes = fread (buffer, 1, EIGHT_KB, src)) > 0)
        fwrite (buffer, 1, bytes, dst);
    }
  else {
    sprintf(msg_buff,"error opening src or dst: %s %s\n",netPtr->fileStructPtr->mtxfile,backupFile);
    LOG_ERROR (msg_buff);
    if (src != NULL) fclose(src);
    if (dst != NULL) fclose(dst);
    pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));
    return FOPENERROR;
  }

  if (src != NULL)
    fclose (src);
  if (dst != NULL)
    fclose (dst);

  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  EXIT(func);
  return ccode;
}


/*********************** WriteMatrixFile ***************************************/
int
WriteMatrixFile (ThreadStruct * netPtr)
{
  int i, j, k;
  int ccode = 0;
  FILE *mfp;
  char msg_buff[EIGHT_KB];

  char func[] = "WriteMatrixFile";
  ENTER (func);

  if ((mfp = fopen (netPtr->fileStructPtr->mtxfile, "w")) == NULL)
    {
      sprintf(msg_buff,"Cannot open matrix file for write: %s\n",netPtr->fileStructPtr->mtxfile);
      LOG_ERROR(msg_buff);
      return FOPENERROR;
    }

  fwrite (&netPtr->netStructPtr->learningRate, sizeof (netPtr->netStructPtr->learningRate), 1, mfp);
  fwrite (&netPtr->netStructPtr->momentum, sizeof (netPtr->netStructPtr->momentum), 1, mfp);
  fwrite (&netPtr->netStructPtr->trainingTolerance, sizeof (netPtr->netStructPtr->trainingTolerance), 1, mfp);
  fwrite (&netPtr->netStructPtr->trainingSuccessPercentage, sizeof (netPtr->netStructPtr->trainingSuccessPercentage), 1, mfp);
  fwrite (&netPtr->netStructPtr->testingTolerance, sizeof (netPtr->netStructPtr->testingTolerance), 1, mfp);
  fwrite (&netPtr->netStructPtr->activationFunction, sizeof (netPtr->netStructPtr->activationFunction), 1, mfp);
  fwrite (&netPtr->netStructPtr->layers, sizeof (netPtr->netStructPtr->layers), 1, mfp);

  for (i = 0; i < MAX_LAYERS; i++)
    {
      fwrite (&(netPtr->netStructPtr->neurons[i]), sizeof (netPtr->netStructPtr->neurons[i]), 1, mfp);
    }

  for (i = 0; i < netPtr->netStructPtr->layers - 1; i++)
    {
      for (j = 0; j < netPtr->netStructPtr->neurons[i]+1; j++)
        {
          fwrite (netPtr->netStructPtr->weights[i][j], sizeof (float),
                  (size_t)netPtr->netStructPtr->neurons[i + 1], mfp);
        }
    }

  fwrite (netPtr->netStructPtr->minInputArray, 1,
          netPtr->netStructPtr->neurons[0] * sizeof (float), mfp);
  fwrite (netPtr->netStructPtr->minOutputArray, 1,
          netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1] * sizeof (float), mfp);
  fwrite (netPtr->netStructPtr->maxInputArray, 1,
          netPtr->netStructPtr->neurons[0] * sizeof (float), mfp);
  fwrite (netPtr->netStructPtr->maxOutputArray, 1,
          netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1] * sizeof (float), mfp);

  fclose (mfp);

  EXIT(func);
  return ccode;
}


/************************** WriteStatsFileHeader ******************************/
int WriteStatsFileHeader(ThreadStruct *netPtr) {

  char *func="WriteStatsFileHeader";
  ENTER(func);

  FILE *outFile=fopen(netPtr->fileStructPtr->stsfile,"w");
  if (outFile == NULL) {
    EXIT(func);
    return FWRITEERROR;
  }

  fprintf(outFile,"Date,Time,FilePass,PassFail,SumSquaredError,PercentFactsCorrect\n");

  fclose(outFile);

  EXIT(func);
  return SUCCESS;
}


/************************** WriteFailedFactsFileHeader ************************/
int WriteFailedFactsFileHeader(ThreadStruct *netPtr) {

  char *func="WriteFailedFactsFileHeader";
  ENTER(func);

  FILE *outFile=fopen(netPtr->fileStructPtr->failedfactsfile,"w");
  if (outFile == NULL) {
    EXIT(func);
    return FWRITEERROR;
  }

  fprintf(outFile,"Date,Time,FilePass,FactNum,SumSquaredError,");

  int i; for(i=0;i<netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers-1];i++) {
    fprintf(outFile,"Output%d,",i);
  }
  for(i=0;i<netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers-1];i++) {
    if (i == netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers-1]-1)
      fprintf(outFile,"ExpectedOutput%d",i);
    else
      fprintf(outFile,"ExpectedOutput%d,",i);
  }

  fprintf(outFile,"\n");
  fclose(outFile);

  EXIT(func);
  return SUCCESS;
}


/************************** WriteOutputFileHeader *****************************/
int WriteOutputFileHeader(ThreadStruct *netPtr, int mode) {

  char *func="WriteOutoutFileHeader";
  ENTER(func);

  FILE *outFile=fopen(netPtr->fileStructPtr->outfile,"w");
  if (outFile == NULL) {
    EXIT(func);
    return FWRITEERROR;
  }

  int i;
  int nIn    = netPtr->netStructPtr->neurons[0];
  int nOut   = netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1];
  int catOut = netPtr->fileStructPtr->categoricalOutput;
  int lblFirst = netPtr->fileStructPtr->outputLabelFirst && catOut;
  int hasExpected = (mode == MODE_TRAINING || mode == MODE_TESTING);

  if (lblFirst) {
    if (hasExpected)
      fprintf(outFile,"PredictedClass,ExpectedClass,");
    else
      fprintf(outFile,"PredictedClass,");
    for (i = 0; i < nIn - 1; i++) fprintf(outFile,"Input%d,",i);
    fprintf(outFile,"Input%d\n", nIn - 1);
  } else {
    for (i = 0; i < nIn; i++) fprintf(outFile,"Input%d,",i);

    if (catOut) {
      if (hasExpected)
        fprintf(outFile,"PredictedClass,ExpectedClass\n");
      else
        fprintf(outFile,"PredictedClass\n");
    } else {
      for (i = 0; i < nOut - 1; i++) fprintf(outFile,"Output%d,",i);
      fprintf(outFile,"Output%d", nOut - 1);
      if (hasExpected) {
        fprintf(outFile,",");
        for (i = 0; i < nOut - 1; i++) fprintf(outFile,"ExpectedOutput%d,",i);
        fprintf(outFile,"ExpectedOutput%d", nOut - 1);
      }
      fprintf(outFile,"\n");
    }
  }

  fclose(outFile);

  EXIT(func);
  return SUCCESS;
}


/************************** WriteStatsFileLine ********************************/
int WriteStatsFileLine(ThreadStruct *netPtr, int filePass, int passFail, float sumSquaredError, float percentCorrect) {

  char *passFailString;
  char *func="WriteStatsFileLine";
  ENTER(func);

  char sdate[1000];
  char stime[1000];
  time_t t = time(NULL);
  struct tm * p = localtime(&t);
  strftime(sdate, 1000, "%Y-%m-%d", p);
  strftime(stime, 1000, "%H:%M:%S", p);

  FILE *outFile=fopen(netPtr->fileStructPtr->stsfile,"a");
  if (outFile == NULL) {
    EXIT(func);
    return FWRITEERROR;
  }

  passFailString = (passFail > 0 ? "pass" : "fail");

  fprintf(outFile,"\"%s\",\"%s\",%d,\"%s\",%f,%f\n", sdate, stime, filePass, passFailString, sumSquaredError, percentCorrect);
  fclose(outFile);

  EXIT(func);
  return SUCCESS;
}


/************************** WriteOutputFileLine *******************************/
int WriteOutputFileLine(ThreadStruct *netPtr, int mode) {

  char *func="WriteOutputFileLine";
  ENTER(func);

  FILE *outFile=fopen(netPtr->fileStructPtr->outfile,"a");
  if (outFile == NULL) {
    EXIT(func);
    return FWRITEERROR;
  }

  int i;
  int nIn    = netPtr->netStructPtr->neurons[0];
  int nOut   = netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1];
  int catOut = netPtr->fileStructPtr->categoricalOutput;
  int lblFirst = netPtr->fileStructPtr->outputLabelFirst && catOut;
  int hasExpected = (mode == MODE_TRAINING || mode == MODE_TESTING);

  int pred = 0, expLabel = 0;
  if (catOut) {
    if (nOut == 1) {
      /* One-hot with 1 neuron: output≈1 → class 0 (inactive), output≈0 → class 1 (active) */
      pred = (netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][0] >= 0.5f) ? 0 : 1;
      if (hasExpected)
        expLabel = (netPtr->netStructPtr->expectedOutputArray[0] >= 0.5f) ? 0 : 1;
    } else {
      float best = netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][0];
      for (i = 1; i < nOut; i++) {
        if (netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][i] > best) {
          best = netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][i];
          pred = i;
        }
      }
      if (hasExpected) {
        for (i = 1; i < nOut; i++)
          if (netPtr->netStructPtr->expectedOutputArray[i] > netPtr->netStructPtr->expectedOutputArray[expLabel])
            expLabel = i;
      }
    }
  }

  if (lblFirst) {
    if (hasExpected)
      fprintf(outFile,"%d,%d,", pred, expLabel);
    else
      fprintf(outFile,"%d,", pred);
    for (i = 0; i < nIn - 1; i++) fprintf(outFile,"%f,", netPtr->netStructPtr->vectors[0][i]);
    fprintf(outFile,"%f", netPtr->netStructPtr->vectors[0][nIn - 1]);
  } else {
    for (i = 0; i < nIn; i++) fprintf(outFile,"%f,", netPtr->netStructPtr->vectors[0][i]);

    if (catOut) {
      if (hasExpected)
        fprintf(outFile,"%d,%d", pred, expLabel);
      else
        fprintf(outFile,"%d", pred);
    } else {
      for (i = 0; i < nOut - 1; i++)
        fprintf(outFile,"%f,", netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][i]);
      fprintf(outFile,"%f", netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][nOut - 1]);
      if (hasExpected) {
        fprintf(outFile,",");
        for (i = 0; i < nOut - 1; i++) fprintf(outFile,"%f,", netPtr->netStructPtr->expectedOutputArray[i]);
        fprintf(outFile,"%f", netPtr->netStructPtr->expectedOutputArray[nOut - 1]);
      }
    }
  }

  fprintf(outFile,"\n");
  fclose(outFile);

  EXIT(func);
  return SUCCESS;
}


/************************** WriteFailedFactsFileLine **************************/
int WriteFailedFactsFileLine(ThreadStruct *netPtr, int factNumber, float sumSquaredErrorThisFact) {

  char *func="WriteFailedFactsFileLine";
  ENTER(func);

  char sdate[1000];
  char stime[1000];
  time_t t = time(NULL);
  struct tm * p = localtime(&t);
  strftime(sdate, 1000, "%Y-%m-%d", p);
  strftime(stime, 1000, "%H:%M:%S", p);

  FILE *outFile=fopen(netPtr->fileStructPtr->failedfactsfile,"a");
  if (outFile == NULL) {
    EXIT(func);
    return FWRITEERROR;
  }

  fprintf(outFile,"\"%s\",\"%s\",%d,%d,%f,", sdate, stime, netPtr->netStructPtr->numberOfPasses, factNumber, sumSquaredErrorThisFact);

  int i;
  if (netPtr->fileStructPtr->categoricalOutput) {
    int numOut = netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers - 1];
    int pred = 0, expLabel = 0;
    if (numOut == 1) {
      pred     = (netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][0] >= 0.5f) ? 0 : 1;
      expLabel = (netPtr->netStructPtr->expectedOutputArray[0] >= 0.5f) ? 0 : 1;
    } else {
      for (i = 1; i < numOut; i++) {
        if (netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][i] >
            netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers - 1][pred]) pred = i;
        if (netPtr->netStructPtr->expectedOutputArray[i] >
            netPtr->netStructPtr->expectedOutputArray[expLabel]) expLabel = i;
      }
    }
    fprintf(outFile,"%d,%d", pred, expLabel);
  } else {
    for(i=0;i<netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers-1];i++) {
      fprintf(outFile,"%f,",netPtr->netStructPtr->vectors[netPtr->netStructPtr->layers-1][i]);
    }
    for(i=0;i<netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers-1];i++) {
      if (i == netPtr->netStructPtr->neurons[netPtr->netStructPtr->layers-1]-1)
        fprintf(outFile,"%f",netPtr->netStructPtr->expectedOutputArray[i]);
      else
        fprintf(outFile,"%f,",netPtr->netStructPtr->expectedOutputArray[i]);
    }
  }

  fprintf(outFile,"\n");
  fclose(outFile);

  EXIT(func);
  return SUCCESS;
}
