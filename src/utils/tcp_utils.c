#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <ifaddrs.h>

/* Replace NaN or Inf with 0 so cJSON always emits valid JSON */
#define SAFE_FLOAT(x) (isfinite(x) ? (x) : 0.0f)
#include "cJSON.h"
#include "log_utils.h"
#include "packet.h"


pthread_mutex_t jSONParserMtx;

extern int   statusFlag;
extern int   debugPackets;
extern int   debugEnterExit;
extern char *networkInterface;



int ProcessPacketRPC(PacketStruct *);
int readBytes(int socket, char *buffer, int bytesToRead, int *bytesRead);
int writeBytes(int socket, char *buffer, int bytesToWrite, int *bytesWritten);
extern int GetStatusFlag(void);




char * GetIPAddress(char * interface) {

  struct ifaddrs *ifaddr, *ifa;
  char buf[INET_ADDRSTRLEN];
  char *ret = NULL;

  /* First try the requested interface name */
  if (getifaddrs(&ifaddr) == 0) {
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == NULL)
        continue;
      if (ifa->ifa_addr->sa_family != AF_INET)
        continue;
      if (strcmp(ifa->ifa_name, interface) != 0)
        continue;
      inet_ntop(AF_INET, &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr,
                buf, sizeof(buf));
      ret = strdup(buf);
      break;
    }

    /* Fall back to the first non-loopback, up-and-running IPv4 address */
    if (ret == NULL) {
      for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
          continue;
        if (ifa->ifa_addr->sa_family != AF_INET)
          continue;
        if (ifa->ifa_flags & IFF_LOOPBACK)
          continue;
        if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING))
          continue;
        inet_ntop(AF_INET, &((struct sockaddr_in *)ifa->ifa_addr)->sin_addr,
                  buf, sizeof(buf));
        ret = strdup(buf);
        break;
      }
    }

    freeifaddrs(ifaddr);
  }

  if (ret == NULL)
    ret = strdup("127.0.0.1");

  return ret;

}


void FreePacket(PacketStruct *packet) {

  char *func="FreePacket";
  ENTER(func);

  if (packet != NULL) {

    if (packet->status != NULL)
      free(packet->status);

    if (packet->src != NULL)
      free(packet->src);

    if (packet->dest != NULL)
      free(packet->dest);
    
    if (packet->mtxfile != NULL)
      free(packet->mtxfile);

    if (packet->infile != NULL)
      free(packet->infile);

    if (packet->outfile != NULL)
      free (packet->outfile);

    if (packet->stsfile != NULL)
      free (packet->stsfile);

    if (packet->failedfactsfile != NULL)
      free (packet->failedfactsfile);


    free (packet);
  }
  
  
  EXIT(func);
  
  
}





PacketStruct *JSONToPacket(char *JSON) {




  char *func="JSONToPacket";
  ENTER(func);

  
  PacketStruct *packet;
  packet = calloc(1,sizeof (PacketStruct));
  if (packet == NULL) {
    LOG_ERROR("Error allocating packet in buildPacket");
    return packet;
  }


  // use cJSON to parse the JSON string
  // this is not thread sare so mutex access to the parser
  pthread_mutex_lock (&jSONParserMtx);
  cJSON *root = cJSON_Parse(JSON);
  pthread_mutex_unlock (&jSONParserMtx);

  if (root == NULL) {
    LOG_ERROR("JSONToPacket: cJSON_Parse returned NULL — returning empty packet");
    packet->status = strdup("");
    packet->src    = strdup("");
    packet->dest   = strdup("");
    packet->mtxfile         = strdup("");
    packet->infile          = strdup("");
    packet->outfile         = strdup("");
    packet->stsfile         = strdup("");
    packet->failedfactsfile = strdup("");
    packet->timestamp       = strdup("");
    return packet;
  }

  cJSON *item;

#define GETINT(field) \
  item = cJSON_GetObjectItem(root, #field); \
  packet->field = item ? item->valueint : 0;

#define GETDBL(field) \
  item = cJSON_GetObjectItem(root, #field); \
  packet->field = item ? (float)item->valuedouble : 0.0f;

#define GETSTR(field) \
  item = cJSON_GetObjectItem(root, #field); \
  packet->field = strdup((item && item->valuestring) ? item->valuestring : "");

  GETINT(netID)
  GETINT(packetType)
  GETINT(ccode)
  GETSTR(status)
  GETSTR(src)
  GETSTR(dest)

  cJSON *neurons = cJSON_GetObjectItem(root, "neurons");
  if (neurons != NULL) {
    int i; for (i=0;i<cJSON_GetArraySize(neurons);i++) {
      cJSON *n = cJSON_GetArrayItem(neurons,i);
      packet->neurons[i] = n->valueint;
    }
  }

  GETINT(layers)
  GETDBL(learningRate)
  GETDBL(momentum)
  GETDBL(trainingTolerance)
  GETDBL(trainingSuccessPercentage)
  GETDBL(testingTolerance)
  GETDBL(nudgeAmount)
  GETSTR(mtxfile)
  GETSTR(infile)
  GETSTR(outfile)
  GETSTR(stsfile)
  GETSTR(failedfactsfile)
  GETINT(activationFunction)
  GETINT(callbackPort)
  GETDBL(weightDecay)
  GETDBL(dropoutRate)
  GETINT(useAdam)
  GETDBL(adamBeta1)
  GETDBL(adamBeta2)
  GETDBL(adamEpsilon)
  GETINT(inputFileHasHeader)
  GETINT(groundTruthFirst)
  GETINT(categoricalOutput)

  item = cJSON_GetObjectItem(root, "outputLabelFirst");
  packet->outputLabelFirst = item ? item->valueint : 1;

  GETDBL(loadAvg)

  item = cJSON_GetObjectItem(root, "memFree");
  packet->memFree = item ? (long)item->valuedouble : 0L;
  item = cJSON_GetObjectItem(root, "swapFree");
  packet->swapFree = item ? (long)item->valuedouble : 0L;

  GETINT(cpuCount)
  GETINT(activeNetworks)
  GETINT(gpuCount)
  item = cJSON_GetObjectItem(root, "gpuVramFreeMB");
  packet->gpuVramFreeMB = item ? (long)item->valueint : 0L;
  GETINT(gpuUtilPct)
  GETINT(gpuComputeSM)

  GETINT(numInputRows)
  GETDBL(meanSumSquaredError)
  GETDBL(percentFactsCorrect)
  GETDBL(mae)
  GETDBL(rmse)
  GETINT(numberOfPasses)

  item = cJSON_GetObjectItem(root, "timestamp");
  packet->timestamp = strdup((item && item->valuestring) ? item->valuestring : "");
  item = cJSON_GetObjectItem(root, "trainingDuration");
  packet->trainingDuration = item ? (long)item->valueint : 0L;
  item = cJSON_GetObjectItem(root, "trainingTimeout");
  packet->trainingTimeout = item ? (long)item->valueint : 0L;

#undef GETINT
#undef GETDBL
#undef GETSTR

  // free the JSON tree struct
  cJSON_Delete(root);
  
  EXIT(func);
  return packet;
}  



char *PacketToJSON(PacketStruct *packet) {
  char *func = "PacketToJSON";

  ENTER(func);

  // initialize any NULL strings to empty strings before parsing
  
  if (packet->status == NULL)
    packet->status = strdup("");

  if (packet->src == NULL)
    packet->src = strdup("");

  if (packet->dest == NULL)
    packet->dest = strdup("");

 
  if (packet->mtxfile == NULL)
    packet->mtxfile = strdup("");

  if (packet->infile == NULL)
    packet->infile = strdup("");

  if (packet->outfile == NULL)
    packet->outfile = strdup("");

  if (packet->stsfile == NULL)
    packet->stsfile = strdup("");

  if (packet->failedfactsfile == NULL)
    packet->failedfactsfile = strdup("");

  

  cJSON *root;
  root=cJSON_CreateObject();

  cJSON_AddNumberToObject(root, "netID", packet->netID);
  cJSON_AddNumberToObject(root, "packetType", packet->packetType);
  cJSON_AddNumberToObject(root, "ccode", packet->ccode);

  cJSON_AddStringToObject(root, "status", packet->status);
  cJSON_AddStringToObject(root, "src", packet->src);
  cJSON_AddStringToObject(root, "dest", packet->dest);

  cJSON_AddItemToObject(root, "neurons", cJSON_CreateIntArray(packet->neurons,MAX_LAYERS));

  
  cJSON_AddNumberToObject(root, "layers", packet->layers);
  cJSON_AddNumberToObject(root, "learningRate", SAFE_FLOAT(packet->learningRate));
  cJSON_AddNumberToObject(root, "momentum", SAFE_FLOAT(packet->momentum));
  cJSON_AddNumberToObject(root, "trainingTolerance", SAFE_FLOAT(packet->trainingTolerance));
  cJSON_AddNumberToObject(root, "trainingSuccessPercentage", SAFE_FLOAT(packet->trainingSuccessPercentage));
  cJSON_AddNumberToObject(root, "testingTolerance", SAFE_FLOAT(packet->testingTolerance));
  cJSON_AddNumberToObject(root, "nudgeAmount", SAFE_FLOAT(packet->nudgeAmount));

  cJSON_AddStringToObject(root, "mtxfile", packet->mtxfile);
  cJSON_AddStringToObject(root, "infile", packet->infile);
  cJSON_AddStringToObject(root, "outfile", packet->outfile);
  cJSON_AddStringToObject(root, "stsfile", packet->stsfile);
  cJSON_AddStringToObject(root, "failedfactsfile", packet->failedfactsfile);
  cJSON_AddNumberToObject(root, "activationFunction", packet->activationFunction);
  cJSON_AddNumberToObject(root, "callbackPort", packet->callbackPort);
  cJSON_AddNumberToObject(root, "weightDecay", SAFE_FLOAT(packet->weightDecay));
  cJSON_AddNumberToObject(root, "dropoutRate", SAFE_FLOAT(packet->dropoutRate));
  cJSON_AddNumberToObject(root, "useAdam", packet->useAdam);
  cJSON_AddNumberToObject(root, "adamBeta1", SAFE_FLOAT(packet->adamBeta1));
  cJSON_AddNumberToObject(root, "adamBeta2", SAFE_FLOAT(packet->adamBeta2));
  cJSON_AddNumberToObject(root, "adamEpsilon", SAFE_FLOAT(packet->adamEpsilon));
  cJSON_AddNumberToObject(root, "inputFileHasHeader", packet->inputFileHasHeader);
  cJSON_AddNumberToObject(root, "groundTruthFirst", packet->groundTruthFirst);
  cJSON_AddNumberToObject(root, "categoricalOutput", packet->categoricalOutput);
  cJSON_AddNumberToObject(root, "outputLabelFirst", packet->outputLabelFirst);

  cJSON_AddNumberToObject(root, "loadAvg", SAFE_FLOAT(packet->loadAvg));
  cJSON_AddNumberToObject(root, "memFree", packet->memFree);
  cJSON_AddNumberToObject(root, "swapFree", packet->swapFree);
  cJSON_AddNumberToObject(root, "cpuCount", packet->cpuCount);
  cJSON_AddNumberToObject(root, "activeNetworks", packet->activeNetworks);
  cJSON_AddNumberToObject(root, "gpuCount", packet->gpuCount);
  cJSON_AddNumberToObject(root, "gpuVramFreeMB", packet->gpuVramFreeMB);
  cJSON_AddNumberToObject(root, "gpuUtilPct", packet->gpuUtilPct);
  cJSON_AddNumberToObject(root, "gpuComputeSM", packet->gpuComputeSM);
  cJSON_AddNumberToObject(root, "numInputRows", packet->numInputRows);
  cJSON_AddNumberToObject(root, "meanSumSquaredError", SAFE_FLOAT(packet->meanSumSquaredError));
  cJSON_AddNumberToObject(root, "percentFactsCorrect", SAFE_FLOAT(packet->percentFactsCorrect));
  cJSON_AddNumberToObject(root, "mae",  SAFE_FLOAT(packet->mae));
  cJSON_AddNumberToObject(root, "rmse", SAFE_FLOAT(packet->rmse));
  cJSON_AddNumberToObject(root, "numberOfPasses", packet->numberOfPasses);
  cJSON_AddStringToObject(root, "timestamp", packet->timestamp ? packet->timestamp : "");
  cJSON_AddNumberToObject(root, "trainingDuration", packet->trainingDuration);
  cJSON_AddNumberToObject(root, "trainingTimeout",  packet->trainingTimeout);


  char *JSON = cJSON_PrintUnformatted(root);


  // debug it
  LOG_DEBUG_PACKET(packet);

  // free the JSON parser tree
  cJSON_Delete(root);


  EXIT(func);

  return JSON;
}





PacketStruct *ProcessPacketClient(char *destIP, int destPort, PacketStruct *packet) {
  int sockfd;
  struct sockaddr_in serv_addr;
  char msg_buff[EIGHT_KB];
  char func[] = "ProcessPacketClient";
  ENTER(func);

  if (packet->src != NULL) free(packet->src);
  packet->src = GetIPAddress(networkInterface);
  if (packet->dest != NULL) free(packet->dest);
  packet->dest = strdup(destIP);

  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = inet_addr(destIP);
  serv_addr.sin_port = htons(destPort);

  if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    LOG_ERROR("Client cannot open socket, returning NULL\n");
    return NULL;
  }

  if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    sprintf(msg_buff, "Client cannot connect to server %s at port %d\n", destIP, destPort);
    LOG_ERROR(msg_buff);
    close(sockfd);
    return NULL;
  }

  char *JSON = PacketToJSON(packet);
  FreePacket(packet);

  int jsonLen = (int)strlen(JSON);
  uint32_t nlen = htonl((uint32_t)jsonLen);
  int dummy;
  if (writeBytes(sockfd, (char *)&nlen, 4, &dummy) != 4 ||
      writeBytes(sockfd, JSON, jsonLen, &dummy) != jsonLen) {
    LOG_ERROR("Client write error on socket\n");
    free(JSON);
    close(sockfd);
    return NULL;
  }
  free(JSON);

  /* Read response: 4-byte length prefix then JSON body */
  uint32_t rlen_net;
  if (readBytes(sockfd, (char *)&rlen_net, 4, &dummy) != 4) {
    LOG_ERROR("Server read error or connection closed before response received\n");
    close(sockfd);
    return NULL;
  }
  int rlen = (int)ntohl(rlen_net);
  if (rlen <= 0 || rlen > 16 * 1024 * 1024) {
    LOG_ERROR("Server response length out of range\n");
    close(sockfd);
    return NULL;
  }
  char *respJSON = calloc(1, (size_t)rlen + 1);
  if (!respJSON) {
    LOG_ERROR("Client cannot allocate response buffer\n");
    close(sockfd);
    return NULL;
  }
  if (readBytes(sockfd, respJSON, rlen, &dummy) != rlen) {
    LOG_ERROR("Server read error: short response\n");
    free(respJSON);
    close(sockfd);
    return NULL;
  }
  close(sockfd);

  PacketStruct *retPacket = JSONToPacket(respJSON);
  free(respJSON);
  LOG_DEBUG_PACKET(retPacket);

  EXIT(func);
  return retPacket;
}




void ProcessPacketServer(int *port) {
  extern int RPCServerSocket;
  socklen_t clilen;
  struct sockaddr_in cli_addr, serv_addr;
  char msg_buff[EIGHT_KB];
  char func[] = "ProcessPacketServer";
  ENTER(func);

  if ((RPCServerSocket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    LOG_ERROR("Server cannot open a socket\n");
    exit(1);
  }

  int reuseopt = 1;
  setsockopt(RPCServerSocket, SOL_SOCKET, SO_REUSEADDR, &reuseopt, sizeof(reuseopt));

  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(*port);

  if (bind(RPCServerSocket, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    LOG_ERROR("Server cannot bind local address to socket\n");
    exit(1);
  }

  clilen = sizeof(cli_addr);
  listen(RPCServerSocket, 10);

  while (GetStatusFlag() == STATUS_RUN) {

    int childSockfd = accept(RPCServerSocket, (struct sockaddr *) &cli_addr, &clilen);
    if (childSockfd < 0) {
      LOG_ERROR("Server accept error\n");
      continue;
    }

    /* Read request: 4-byte length prefix then JSON body */
    uint32_t rlen_net;
    int dummy;
    if (readBytes(childSockfd, (char *)&rlen_net, 4, &dummy) != 4) {
      LOG_ERROR("Server read error on socket\n");
      close(childSockfd);
      continue;
    }
    int rlen = (int)ntohl(rlen_net);
    if (rlen <= 0 || rlen > 16 * 1024 * 1024) {
      LOG_ERROR("Request length out of range\n");
      close(childSockfd);
      continue;
    }
    char *reqJSON = calloc(1, (size_t)rlen + 1);
    if (!reqJSON) {
      LOG_ERROR("Server cannot allocate request buffer\n");
      close(childSockfd);
      continue;
    }
    if (readBytes(childSockfd, reqJSON, rlen, &dummy) != rlen) {
      LOG_ERROR("Server read error: short request\n");
      free(reqJSON);
      close(childSockfd);
      continue;
    }

    PacketStruct *packet = JSONToPacket(reqJSON);
    free(reqJSON);
    LOG_DEBUG_PACKET(packet);

    ProcessPacketRPC(packet);

    free(packet->dest);
    packet->dest = strdup(packet->src);
    free(packet->src);
    packet->src = GetIPAddress(networkInterface);

    char *respJSON = PacketToJSON(packet);
    FreePacket(packet);

    int jsonLen = (int)strlen(respJSON);
    uint32_t nlen = htonl((uint32_t)jsonLen);
    if (writeBytes(childSockfd, (char *)&nlen, 4, &dummy) != 4 ||
        writeBytes(childSockfd, respJSON, jsonLen, &dummy) != jsonLen) {
      LOG_ERROR("Server write error on socket\n");
    }
    free(respJSON);
    close(childSockfd);

  } /* end of while loop */

  shutdown(RPCServerSocket, SHUT_RDWR);
  close(RPCServerSocket);

  EXIT(func);
}


int readBytes(int socket, char *buffer, int bytesToRead, int *bytesRead)
{
    *bytesRead = 0;
    while(*bytesRead < bytesToRead)
    {
        int ret = read(socket, buffer + *bytesRead, bytesToRead - *bytesRead);
        if(ret <= 0)
        {
           /* either connection was closed or an error occurred */
           return ret;
        }
        else
        {
           *bytesRead += ret;
        }
    }
    return *bytesRead;
}



int writeBytes(int socket, char *buffer, int bytesToWrite, int *bytesWritten) {

    *bytesWritten = 0;
    while(*bytesWritten < bytesToWrite)
    {
        int ret = write(socket, buffer + *bytesWritten, bytesToWrite - *bytesWritten);
        if(ret <= 0)
        {
           /* either connection was closed or an error occurred */
           return ret;
        }
        else
        {
           *bytesWritten += ret;
        }
    }

  return *bytesWritten;
}


