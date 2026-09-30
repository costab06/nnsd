

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include "common.h"
#include "packet.h"
#include "log_utils.h"




int debugPackets=FALSE, debugMath=FALSE, debugEnterExit=FALSE, debugLevel1=FALSE, debugLevel2=FALSE, debugWeights=FALSE;

pthread_mutex_t debugfileMtx;

static FILE *output=NULL;

void
log_msg (const char *type, const char *msg)
{
  
  char buff[1024*1024];
  char logfile[1024];
  
  time_t t;
  long old_umask;
  struct stat st_buff;
  
  pthread_mutex_lock (&debugfileMtx);
  
  old_umask=umask(022);
  strcpy (buff, type);
  
  time (&t);
  strcat (buff, (char *) ctime (&t));
  
  
  strcat (buff, "\t");
  strcat (buff, msg);
  strcat (buff, "\n\n");
  
  
  strcpy(logfile,"/opt/neural/logs/nns.log");
  
  output = fopen (logfile,"a");
  
  if (output == NULL)
    
    // write to stderr if can't write to log file
    fprintf(stderr,"%s",buff);
  else
    {
      // write to the specified log file
      fprintf (output,"%s",buff);
      fflush(output);
      fclose(output);
      output = NULL;
    }
  umask(old_umask);
  pthread_mutex_unlock (&debugfileMtx);
}




void
log_debug1 (const char *msg)
{
  log_msg ("DEBUG_LEVEL1:\t", msg);
}


void
log_debug2 (const char *msg)
{
  log_msg ("DEBUG_LEVEL2:\t", msg);
}


void
log_packet (const char *type, const PacketStruct *packet)
{
  char msg_buff[8192];
  char tmp[8192];
  
  time_t t;
  long old_umask;
  struct stat st_buff;
  
  if (packet != NULL) {

    pthread_mutex_lock (&debugfileMtx);
    
    old_umask=umask(022);
    strcpy (msg_buff, type);
    
    time (&t);
    strcat (msg_buff, (char *) ctime (&t));
    
    strcat (msg_buff, "\t");
    
    strcat (msg_buff, "Packet: \n");
    
    sprintf(tmp,"netID: %d\n", packet->netID);
    strcat(msg_buff,tmp);
    
    sprintf(tmp, "packetType: %d\n", packet->packetType);
    strcat (msg_buff,tmp);
    
    sprintf(tmp, "ccode: %d\n", packet->ccode);
    strcat (msg_buff,tmp);
    
    
    sprintf(tmp, "status: %s\n", packet->status);
    strcat (msg_buff,tmp);
    
    sprintf(tmp, "src: %s\n", packet->src);
    strcat (msg_buff,tmp);
    
    sprintf(tmp, "dest: %s\n", packet->dest);
    strcat (msg_buff,tmp);
    
    sprintf(tmp, "neurons: %d %d %d %d %d %d %d %d\n", 
	    packet->neurons[0],packet->neurons[1],
	    packet->neurons[2],packet->neurons[3],
	    packet->neurons[4],packet->neurons[5],
	    packet->neurons[6],packet->neurons[7]);
    strcat (msg_buff,tmp);
    
    sprintf(tmp, "layers: %d\n", packet->layers);
    strcat (msg_buff,tmp);
    
    sprintf(tmp, "learningRate: %f\n", packet->learningRate);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "momentum: %f \n", packet->momentum);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "trainingTolerance: %f \n", packet->trainingTolerance);
    strcat (msg_buff, tmp);

    sprintf(tmp, "trainingSuccessPercentage: %f \n", packet->trainingSuccessPercentage);
    strcat (msg_buff, tmp);

    sprintf(tmp, "testingTolerance: %f \n", packet->testingTolerance);
    strcat (msg_buff, tmp);

    
    sprintf(tmp, "nudgeAmount: %f\n", packet->nudgeAmount);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "mtxfile: %s\n", packet->mtxfile);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "infile: %s\n", packet->infile);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "outfile: %s\n", packet->outfile);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "stsfile: %s\n", packet->stsfile);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "failedfactsfile: %s\n", packet->failedfactsfile);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "loadAvg: %f\n", packet->loadAvg);
    strcat (msg_buff,tmp);
    
    sprintf(tmp, "memFree: %ld\n", packet->memFree);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "swapFree: %ld\n", packet->swapFree);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "numInputRows: %d\n", packet->numInputRows);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "meanSumSquaredError: %f\n", packet->meanSumSquaredError);
    strcat (msg_buff, tmp);

    sprintf(tmp, "percentFactsCorrect: %f\n", packet->percentFactsCorrect);
    strcat (msg_buff, tmp);
    
    sprintf(tmp, "numberOfPasses: %d\n", packet->numberOfPasses);
    strcat (msg_buff, tmp);

    sprintf(tmp, "trainingTimeout: %ld\n", packet->trainingTimeout);
    strcat (msg_buff, tmp);

    sprintf(tmp, "useAdam: %d adamBeta1: %f adamBeta2: %f adamEpsilon: %f\n",
            packet->useAdam, packet->adamBeta1, packet->adamBeta2, packet->adamEpsilon);
    strcat (msg_buff, tmp);

    strcat (msg_buff, "\n\n");
    
    output = fopen ("/opt/neural/logs/nns.log","a");
    
    if (output == NULL)
      
      // write to stderr if can't write to log file
      fprintf(stderr,"%s",msg_buff);
    else
      {
	// write to the specified log file
	fprintf (output,"%s",msg_buff);
	fflush(output);
	fclose(output);
	output = NULL;
      }
    umask(old_umask);
    
    pthread_mutex_unlock (&debugfileMtx);
  } else {
    sprintf(msg_buff,"ERROR: log_packet called with NULL packet\n");
    LOG_ERROR(msg_buff);
  }
}
  




void close_log()
{
  if (output != NULL)
    {
      fflush(output);
      fclose(output);
      output=NULL;
    }
}







