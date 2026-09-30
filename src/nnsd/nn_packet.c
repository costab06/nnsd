/***************************************************************************
 *                      NNSD Packet Processing                              *
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

#include "common.h"
#include "log_utils.h"
#include "nn_globals.h"
#include "nn_network.h"
#include "nn_sysinfo.h"
#include "nn_training.h"
#include "nn_packet.h"
#include "packet.h"

/* ProcessPacketClient and FreePacket are defined in tcp_utils.c */
extern PacketStruct *ProcessPacketClient(char *, int, PacketStruct *);
extern void FreePacket(PacketStruct *);


/************************** ValidatePacket *******************************/
int ValidatePacket(PacketStruct *packet) {

  char *func = "ValidatePacket";
  int ccode = SUCCESS;

  ENTER(func);

  if (packet->packetType < COEF_PACKET || packet->packetType > REMOVENETWORK_PACKET ||
      packet->packetType == STATS_PACKET ||
      packet->packetType == ERROR_PACKET ||
      packet->packetType == NETFINISHED_PACKET) {
    ccode=PACKETTYPEERROR;
  }

  if ((packet->packetType == RUN_PACKET)
      || (packet->packetType == TRAIN_PACKET)
      || (packet->packetType == TEST_PACKET)
      || (packet->packetType == RETRAIN_PACKET)) {

    if (access (packet->infile, R_OK))
      {
        ccode = FILENOTEXISTERROR;
      }
  }

  EXIT(func);
  return ccode;
}


/*********************** ProcessPacketRPC  ********************************/
int
ProcessPacketRPC (PacketStruct * packet)
{
  ThreadStruct *netPtr;
  char msg_buff[EIGHT_KB];
  char func[] = "ProcessPacketRPC";
  ENTER (func);

  umask (0133);

  LOG_DEBUG_PACKET(packet);

  /* Prepend mountPoint to all file paths in the packet */
  char *tmp;

  tmp = malloc(strlen(mountPoint)+strlen(packet->mtxfile)+1);
  tmp = strcpy(tmp,mountPoint);
  tmp = strcat(tmp,packet->mtxfile);
  free(packet->mtxfile);
  packet->mtxfile = tmp;

  tmp = malloc(strlen(mountPoint)+strlen(packet->infile)+1);
  tmp = strcpy(tmp,mountPoint);
  tmp = strcat(tmp,packet->infile);
  free(packet->infile);
  packet->infile = tmp;

  tmp = malloc(strlen(mountPoint)+strlen(packet->outfile)+1);
  tmp = strcpy(tmp,mountPoint);
  tmp = strcat(tmp,packet->outfile);
  free(packet->outfile);
  packet->outfile = tmp;

  tmp = malloc(strlen(mountPoint)+strlen(packet->stsfile)+1);
  tmp = strcpy(tmp,mountPoint);
  tmp = strcat(tmp,packet->stsfile);
  free(packet->stsfile);
  packet->stsfile = tmp;

  tmp = malloc(strlen(mountPoint)+strlen(packet->failedfactsfile)+1);
  tmp = strcpy(tmp,mountPoint);
  tmp = strcat(tmp,packet->failedfactsfile);
  free(packet->failedfactsfile);
  packet->failedfactsfile = tmp;

  int ccode = ValidatePacket(packet);

  if ( ccode == SUCCESS) {
    switch (packet->packetType)
      {
      case COEF_PACKET:
        packet->ccode = Coef (packet);
        break;

      case RUN_PACKET: {
        if (GetNetwork(packet->netID) != NULL) {
          ccode = NETALREADYPROCESSINGERROR;
        } else {
          if (access (packet->mtxfile, R_OK) == 0) {
            ccode = InitializeNetworkFromMatrixFile (packet, &netPtr);
          } else {
            ccode = FOPENERROR;
          }
          if (ccode != SUCCESS)
            return ccode;
          pthread_create (&(netPtr->thread), (pthread_attr_t *) NULL,
                          RunNetwork, (void *) netPtr);
        }
        break;
      }

      case TRAIN_PACKET: {
        if (GetNetwork(packet->netID) != NULL) {
          ccode = NETALREADYPROCESSINGERROR;
        } else {
          if (access (packet->mtxfile, R_OK) == 0) {
            ccode = InitializeNetworkFromMatrixFile (packet, &netPtr);
          } else {
            ccode = InitializeNetworkFromPacket (packet, &netPtr);
          }
          if (ccode != SUCCESS)
            return ccode;
          pthread_create (&(netPtr->thread), (pthread_attr_t *) NULL,
                          TrainNetwork, (void *) netPtr);
        }
        break;
      }

      case RETRAIN_PACKET: {
        if (GetNetwork(packet->netID) != NULL) {
          ccode = NETALREADYPROCESSINGERROR;
        } else {
          if (access (packet->mtxfile, R_OK) == 0) {
            remove(packet->mtxfile);
          }
          ccode = InitializeNetworkFromPacket (packet, &netPtr);
          if (ccode != SUCCESS)
            return ccode;
          pthread_create (&(netPtr->thread), (pthread_attr_t *) NULL,
                          TrainNetwork, (void *) netPtr);
        }
        break;
      }

      case TEST_PACKET: {
        if (GetNetwork(packet->netID) != NULL) {
          ccode = NETALREADYPROCESSINGERROR;
        } else {
          if (access (packet->mtxfile, R_OK) == 0) {
            ccode = InitializeNetworkFromMatrixFile (packet, &netPtr);
          } else {
            ccode = FOPENERROR;
          }
          if (ccode != SUCCESS)
            return ccode;
          netPtr->netStructPtr->testingTolerance = packet->testingTolerance;
          pthread_create (&(netPtr->thread), (pthread_attr_t *) NULL,
                          TestNetwork, (void *) netPtr);
        }
        break;
      }

      case CANCEL_PACKET: {
        ThreadStruct *temp = GetNetwork(packet->netID);
        if (temp != NULL)
          {
            pthread_mutex_lock (&(temp->threadInfoHoldMtx));
            temp->statusFlag = STATUS_SHUTDOWN;
            pthread_mutex_unlock (&(temp->threadInfoHoldMtx));
          }
        else {
          packet->ccode = NETNOTFOUNDERROR;
        }
        break;
      }

      case NUDGE_PACKET: {
        ThreadStruct *temp = GetNetwork(packet->netID);
        if (temp != NULL)
          {
            pthread_mutex_lock (&(temp->threadInfoHoldMtx));
            temp->nudgeAmount = packet->nudgeAmount;
            pthread_mutex_unlock (&(temp->threadInfoHoldMtx));
          }
        else {
          packet->ccode = NETNOTFOUNDERROR;
        }
        break;
      }

      case SHUTDOWN_PACKET: {
        packet->ccode = SetStatusFlag(STATUS_SHUTDOWN);
        break;
      }

      case REMOVENETWORK_PACKET: {
        if (GetNetwork(packet->netID) != NULL) {
          ccode = NETALREADYPROCESSINGERROR;
        } else {
          if (packet->mtxfile[0] != '\0') remove(packet->mtxfile);
          if (packet->outfile[0]  != '\0') remove(packet->outfile);
          if (packet->stsfile[0]  != '\0') remove(packet->stsfile);
          if (packet->failedfactsfile[0] != '\0') remove(packet->failedfactsfile);
          /* Also remove the backup matrix file (.bak) if present */
          if (packet->mtxfile[0] != '\0') {
            char bakfile[ONE_KB + 4];
            snprintf(bakfile, sizeof(bakfile), "%s", packet->mtxfile);
            char *bdot = strrchr(bakfile, '.');
            if (bdot) *bdot = '\0';
            strncat(bakfile, ".bak", sizeof(bakfile) - strlen(bakfile) - 1);
            remove(bakfile);
          }
          packet->ccode = SUCCESS;
        }
        break;
      }

      default: {
        packet->ccode = PACKETTYPEERROR;
        break;
      }
      }
  } else {
    const char *reason;
    switch (ccode) {
      case FILENOTEXISTERROR: reason = "input file not found or not readable"; break;
      case PACKETTYPEERROR:   reason = "unknown packet type";                  break;
      default:                reason = "packet validation failed";             break;
    }
    sprintf(msg_buff, "Error: invalid packet received: ccode: %d (%s). \n", ccode, reason);
    LOG_ERROR(msg_buff);
    LOG_DEBUG_PACKET(packet);
    if (packet->status) free(packet->status);
    packet->status = strdup(reason);
    packet->ccode = ccode;  /* return the specific ccode, not the generic wrapper */
  }

  EXIT(func);
  return packet->ccode;
}


/******************** CreatePacketFromThreadStruct ***********************/
PacketStruct *CreatePacketFromThreadStruct(ThreadStruct *netPtr, int packetType, int ccode, char *status) {
  char msg_buff[EIGHT_KB];
  char *func = "CreatePacketFromThreadStruct";
  ENTER(func);

  sprintf(msg_buff,"Createing a packet for netID: %d",netPtr->netID);

  PacketStruct *packet = calloc(1,sizeof(PacketStruct));
  if (packet == NULL) {
    return NULL;
  }

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));

  packet->netID = netPtr->netID;
  packet->packetType = packetType;
  packet->ccode = ccode;
  packet->status = strdup(status);

  int i; for(i=0;i<MAX_LAYERS;i++) {
    packet->neurons[i] =  netPtr->netStructPtr->neurons[i];
  }

  packet->layers = netPtr->netStructPtr->layers;
  packet->learningRate = netPtr->netStructPtr->learningRate;
  packet->momentum = netPtr->netStructPtr->momentum;
  packet->trainingTolerance = netPtr->netStructPtr->trainingTolerance;
  packet->trainingSuccessPercentage = netPtr->netStructPtr->trainingSuccessPercentage;
  packet->testingTolerance = netPtr->netStructPtr->testingTolerance;
  packet->nudgeAmount = netPtr->nudgeAmount;

  packet->mtxfile = strdup(netPtr->fileStructPtr->mtxfile);
  packet->infile = strdup(netPtr->fileStructPtr->infile);
  packet->outfile = strdup(netPtr->fileStructPtr->outfile);
  packet->stsfile = strdup(netPtr->fileStructPtr->stsfile);
  packet->failedfactsfile = strdup(netPtr->fileStructPtr->failedfactsfile);

  packet->loadAvg = 0.0f;
  packet->memFree = 0;
  packet->swapFree = 0;

  packet->numInputRows = netPtr->fileStructPtr->numInputRows;
  packet->meanSumSquaredError = netPtr->netStructPtr->meanSumSquaredError;
  packet->percentFactsCorrect = netPtr->netStructPtr->percentFactsCorrect;
  packet->numberOfPasses = netPtr->netStructPtr->numberOfPasses;
  packet->mae  = netPtr->netStructPtr->mae;
  packet->rmse = netPtr->netStructPtr->rmse;

  time_t now = time(NULL);
  struct tm *tm_info = localtime(&now);
  char tsbuf[32];
  strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", tm_info);
  packet->timestamp = strdup(tsbuf);
  packet->trainingDuration = (long)difftime(now, netPtr->netStructPtr->trainingStartTime);

  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  EXIT(func);
  return packet;
}


/************************** SendStatsPacket ******************************/
int SendStatsPacket(ThreadStruct *netPtr) {

  int ccode = SUCCESS;
  char msg_buff[EIGHT_KB];
  char *func="SendStatsPacket";
  ENTER(func);

  PacketStruct *packet = CreatePacketFromThreadStruct(netPtr,STATS_PACKET,SUCCESS,"Success");

  LOG_DEBUG_PACKET(packet);

  PacketStruct *retPacket = ProcessPacketClient (netPtr->src, netPtr->callbackPort, packet);

  if (retPacket != NULL) {
    LOG_DEBUG_PACKET(retPacket);
    FreePacket(retPacket);
  } else {
    sprintf(msg_buff,"Received a null packet from ProcessPacketClient, check connectivity\n");
    LOG_ERROR(msg_buff);
  }

  EXIT(func);
  return (ccode);
}


/************************** SendNetFinishedPacket ************************/
int SendNetFinishedPacket(ThreadStruct *netPtr) {

  int ccode = SUCCESS;
  char *func="SendNetFinishedPacket";
  ENTER(func);

  PacketStruct *packet = CreatePacketFromThreadStruct(netPtr,NETFINISHED_PACKET,SUCCESS,"Success");

  LOG_DEBUG_PACKET(packet);

  PacketStruct *retPacket = ProcessPacketClient (netPtr->src, netPtr->callbackPort, packet);

  if (retPacket != NULL) {
    LOG_DEBUG_PACKET(retPacket);
    FreePacket(retPacket);
  } else {
    LOG_ERROR("SendNetFinishedPacket: received NULL from ProcessPacketClient\n");
  }

  EXIT(func);
  return (ccode);
}
