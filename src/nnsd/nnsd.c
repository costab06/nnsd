/***************************************************************************
 *                        NNSD Runtime — Main / Startup                     *
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

#include "common.h"
#include "log_utils.h"
#include "config.h"
#include "rest_server.h"
#include "nn_globals.h"
#include "nn_network.h"
#include "nn_packet.h"
#include "nn_sysinfo.h"

/* ProcessPacketServer, StartDaemon, GetIPAddress defined in tcp_utils.c / daemon.c */
extern void ProcessPacketServer(int *);
extern void StartDaemon(int, int *);
extern char *GetIPAddress(char *);

/* ── Global variable definitions ──────────────────────────────────────── */

int             RPCServerSocket;
int             statusFlag          = STATUS_RUN;

/* Startup options */
int             runAsDaemon         = FALSE;
char           *mountPoint;
char           *networkInterface;
int             writeTrainingFailedFacts = FALSE;
int             restPort            = NNSD_REST_PORT;

char           *ipAddress;

/* Thread list */
ThreadStruct   *threadInfoList      = NULL;
pthread_mutex_t threadListMtx;

/* Mutexes */
static pthread_mutex_t statusFlagMtx;

/* Defined in log_utils.c */
extern pthread_mutex_t debugfileMtx;
/* Defined in tcp_utils.c (JSON parser serialisation) */
extern pthread_mutex_t jSONParserMtx;


/* Forward declarations */
int  GetOptions(int argc, char *argv[]);
int  InitMutexes(void);
void DestroyMutexes(void);

/************************** main  *******************************************/
int
main (int argc, char *argv[])
{
  int ccode = SUCCESS, port = NNSD_TCP_PORT;
  char func[] = "main";
  char msg_buff[EIGHT_KB];

  sprintf(msg_buff,"=========================================================================\n\tNNSD starting");
  LOG_INFO(msg_buff);

  ENTER (func);

  srand((unsigned int)time(NULL));

  mountPoint        = calloc(1, ONE_KB);
  networkInterface  = calloc(1, ONE_KB);

  GetOptions (argc, argv);

  ipAddress = GetIPAddress(networkInterface);

  printf ("=========================================================================\n");
  printf (" NNSD - Neural Network Service Daemon\n");
  printf ("=========================================================================\n");
  printf ("  Interface    : %s\n", networkInterface[0] ? networkInterface : "(auto)");
  printf ("  IP address   : %s\n", ipAddress);
  printf ("  Port         : %d\n", NNSD_TCP_PORT);
  printf ("  REST port    : %d\n", restPort);
  printf ("  Callback port: %d\n", NNCD_TCP_PORT);
  printf ("  Mount point  : %s\n", mountPoint);
  PrintGpuInfo ();
  printf ("  Daemon mode  : %s\n", runAsDaemon             ? "true" : "false");
  printf ("  Failed facts : %s\n", writeTrainingFailedFacts ? "true" : "false");
  printf ("  debugPackets : %s\n", debugPackets             ? "true" : "false");
  printf ("  debugMath    : %s\n", debugMath                ? "true" : "false");
  printf ("  debugEnterExit: %s\n", debugEnterExit          ? "true" : "false");
  printf ("  debugLevel1  : %s\n", debugLevel1              ? "true" : "false");
  printf ("  debugLevel2  : %s\n", debugLevel2              ? "true" : "false");
  printf ("  debugWeights : %s\n", debugWeights             ? "true" : "false");
  printf ("=========================================================================\n");
  printf (" Log file      : /opt/neural/logs/nns.log\n");
  printf ("=========================================================================\n");
  fflush (stdout);

  ccode = InitMutexes ();
  if (!ccode)
    {
      pthread_t restThread;
      pthread_create(&restThread, NULL, RESTServerThread, &restPort);
      pthread_detach(restThread);

      if (runAsDaemon == FALSE)
        {
          ProcessPacketServer (&port);
        }
      else
        {
          StartDaemon (FALSE, &port);
        }

      ShutdownNetworks ();
    }

  DestroyMutexes ();

  free(mountPoint);
  free(networkInterface);
  free(ipAddress);

  EXIT (func);
}


/************************ InitMutexes **********************************/
int
InitMutexes ()
{
  int ccode = SUCCESS;

  pthread_mutex_init (&threadListMtx, (pthread_mutexattr_t *) NULL);
  pthread_mutex_init (&debugfileMtx, (pthread_mutexattr_t *) NULL);
  pthread_mutex_init (&statusFlagMtx, (pthread_mutexattr_t *) NULL);
  pthread_mutex_init (&jSONParserMtx, (pthread_mutexattr_t *) NULL);

  return ccode;
}


/************************** DestroyMutexes ********************************/
void
DestroyMutexes ()
{
  char func[] = "DestroyMutexes";
  ENTER (func);

  pthread_mutex_destroy (&threadListMtx);
  pthread_mutex_destroy (&debugfileMtx);
  pthread_mutex_destroy (&statusFlagMtx);
  pthread_mutex_destroy (&jSONParserMtx);

  EXIT (func);
}


/************************ GetOptions ****************************/
int
GetOptions (int argc, char *argv[])
{
  int i;
  char func[] = "GetOptions";
  char msg_buff[EIGHT_KB];
  const char *configFile = DEFAULT_CONFIG_FILE;

  ENTER (func);

  strcpy (mountPoint, "/opt/neural/data");
  networkInterface[0] = '\0';
  restPort = NNSD_REST_PORT;

  for (i = 0; i < argc - 1; i++)
    {
      if (!strcmp (argv[i], "-config"))
        {
          configFile = argv[i + 1];
          break;
        }
    }

  ReadConfig (configFile);

  for (i = 0; i < argc; i++)
    {
      if (!strcmp (argv[i], "-runAsDaemon"))
        runAsDaemon = TRUE;

      if (!strcmp (argv[i], "-mountPoint") && i + 1 < argc)
        strcpy (mountPoint, argv[i + 1]);

      if (!strcmp (argv[i], "-networkInterface") && i + 1 < argc)
        strcpy (networkInterface, argv[i + 1]);

      if (!strcmp (argv[i], "-writeTrainingFailedFacts"))
        writeTrainingFailedFacts = TRUE;

      if (!strcmp (argv[i], "-debugPackets"))
        debugPackets = TRUE;

      if (!strcmp (argv[i], "-debugMath"))
        debugMath = TRUE;

      if (!strcmp (argv[i], "-debugEnterExit"))
        debugEnterExit = TRUE;

      if (!strcmp (argv[i], "-debugWeights"))
        debugWeights = TRUE;

      if (!strcmp (argv[i], "-restPort") && i + 1 < argc)
        restPort = atoi (argv[i + 1]);
    }

  sprintf (msg_buff,
           "Options:\n\trunAsDaemon=%s\n\tmountPoint=%s\n\tnetworkInterface=%s\n"
           "\twriteTrainingFailedFacts=%s\n"
           "\tdebugPackets=%s\n\tdebugMath=%s\n\tdebugEnterExit=%s\n\tdebugLevel1=%s\n\tdebugLevel2=%s\n\tdebugWeights=%s\n",
           runAsDaemon             ? "true" : "false",
           mountPoint,
           networkInterface[0]     ? networkInterface : "(auto)",
           writeTrainingFailedFacts ? "true" : "false",
           debugPackets            ? "true" : "false",
           debugMath               ? "true" : "false",
           debugEnterExit          ? "true" : "false",
           debugLevel1             ? "true" : "false",
           debugLevel2             ? "true" : "false",
           debugWeights            ? "true" : "false");
  LOG_INFO (msg_buff);

  EXIT (func);
  return SUCCESS;
}


/****************************SetStatusFlag***************************/
int
SetStatusFlag (int status)
{
  char func[] = "SetStatusFlag";
  ENTER (func);

  pthread_mutex_lock (&statusFlagMtx);
  statusFlag = status;
  pthread_mutex_unlock (&statusFlagMtx);

  EXIT(func);
  return SUCCESS;
}


/****************************GetStatusFlag***************************/
int
GetStatusFlag ()
{
  char func[] = "GetStatusFlag";
  ENTER (func);

  pthread_mutex_lock (&statusFlagMtx);
  int status = statusFlag;
  pthread_mutex_unlock (&statusFlagMtx);

  EXIT(func);
  return status;
}


/****************************SetNetStatusFlag***************************/
int
SetNetStatusFlag (ThreadStruct *netPtr, int status)
{
  char func[] = "SetNetStatusFlag";
  ENTER (func);

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));
  netPtr->statusFlag = status;
  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  EXIT(func);
  return SUCCESS;
}


/****************************GetNetStatusFlag***************************/
int
GetNetStatusFlag (ThreadStruct *netPtr)
{
  char func[] = "GetNetStatusFlag";
  ENTER (func);

  pthread_mutex_lock (&(netPtr->threadInfoHoldMtx));
  int status = netPtr->statusFlag;
  pthread_mutex_unlock (&(netPtr->threadInfoHoldMtx));

  EXIT(func);
  return status;
}
