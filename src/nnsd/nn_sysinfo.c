/***************************************************************************
 *                          NNSD System Info                                *
 **************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef USE_CUDA
#include <nvml.h>
#endif

#include "common.h"
#include "log_utils.h"
#include "nn_globals.h"
#include "nn_sysinfo.h"


/****************************** GetMemFree ******************************/
/* Reports MemAvailable (not MemFree) — this includes reclaimable
 * buffer/page-cache that the kernel will release under memory pressure,
 * which is what "free -m" shows as "available" and what actually matters
 * when deciding whether a new training job can run.               */
int
GetMemFree (long *free)
{
  char func[] = "GetMemFree";

  int ccode = SUCCESS;
  char mem_avail_string[] = "MemAvailable:";
  FILE *mfile;
  char line[1024];
  long mf = 0;
  char msg_buff[EIGHT_KB];

  char *token, *saveptr1;

  ENTER (func);

  if ((mfile = fopen ("/proc/meminfo", "r")) != NULL)
    {
      while (fgets(line, sizeof line, mfile) != NULL)
        {
          if(strstr(line, mem_avail_string) != NULL) {
            token = strtok_r(line, " ", &saveptr1);
            token = strtok_r(saveptr1, " ", &saveptr1);
            mf = atol(token);
          }
        }

      fclose (mfile);
      *free = mf;

      sprintf(msg_buff,"GetMemFree (MemAvailable) found: %ld",mf);
      LOG_DEBUG_LEVEL2(msg_buff);
    }
  else
    {
      LOG_ERROR ("Unable to open /proc/meminfo to calculate memory usage");
      ccode = FREADERROR;
    }

  EXIT (func);
  return ccode;
}


/****************************** GetSwapFree ******************************/
int
GetSwapFree (long *free)
{
  char func[] = "GetSwapFree";

  int ccode = SUCCESS;
  char mem_free_string[] = "SwapFree:";
  FILE *mfile;
  char line[1024];
  long sf = 0;
  char msg_buff[EIGHT_KB];

  char *token, *saveptr1;

  ENTER (func);

  if ((mfile = fopen ("/proc/meminfo", "r")) != NULL)
    {
      while (fgets(line, sizeof line, mfile) != NULL)
        {
          if(strstr(line, mem_free_string) != NULL) {
            token = strtok_r(line, " ", &saveptr1);
            token = strtok_r(saveptr1, " ", &saveptr1);
            sf = atoi(token);
          }
        }

      fclose (mfile);
      *free = sf;

      sprintf(msg_buff,"GetSwapFree found: %ld",sf);
    }
  else
    {
      LOG_ERROR ("Unable to open /proc/meminfo to calculate swap usage");
      ccode = FREADERROR;
    }

  EXIT (func);
  return ccode;
}


/****************************** GetLoadAvg ******************************/
/* Weighted average: weight recent data most heavily.
 * l1 = 1-min, l2 = 5-min, l3 = 15-min.
 * Weights: 3 × 1-min + 2 × 5-min + 1 × 15-min  (sum = 6)
 * A server that just spiked will score high; one that was busy but is
 * now cooling down will score progressively lower.                   */
int
GetLoadAvg (float *load)
{
  int ccode = SUCCESS;
  FILE *lfile;
  float l1, l2, l3;
  char msg_buff[EIGHT_KB];

  char func[] = "GetLoadAvg";

  ENTER (func);

  if ((lfile = fopen ("/proc/loadavg", "r")) != NULL)
    {
      fscanf (lfile, "%f %f %f", &l1, &l2, &l3);
      fclose (lfile);
      *load = (3.0f * l1 + 2.0f * l2 + 1.0f * l3) / 6.0f;

      sprintf(msg_buff,"GetLoadAvg found: %f",*load);
      LOG_DEBUG_LEVEL2(msg_buff);
    }
  else
    {
      LOG_ERROR ("Unable to open /proc/loadavg to calculate current load");
      ccode = FREADERROR;
    }
  EXIT (func);
  return ccode;
}


/****************************** GetCpuCount *****************************/
static int
GetCpuCount (void)
{
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? (int)n : 1;
}


/************************** GetActiveNetworkCount ***********************/
static int
GetActiveNetworkCount (void)
{
  int count = 0;
  pthread_mutex_lock(&threadListMtx);
  ThreadStruct *t = threadInfoList;
  while (t != NULL) { count++; t = t->flink; }
  pthread_mutex_unlock(&threadListMtx);
  return count;
}


#ifdef USE_CUDA
/****************************** GetGpuMetrics ***************************/
/* Queries all CUDA-capable GPUs via NVML and returns metrics for the
 * least-loaded device (the one with the most free VRAM).             */
static void
GetGpuMetrics (int *gpuCount, long *vramFreeMB, int *utilPct, int *computeSM)
{
  *gpuCount   = 0;
  *vramFreeMB = 0;
  *utilPct    = 0;
  *computeSM  = 0;

  if (nvmlInit_v2() != NVML_SUCCESS)
    return;

  unsigned int n = 0;
  nvmlDeviceGetCount_v2(&n);
  *gpuCount = (int)n;

  for (unsigned int i = 0; i < n; i++) {
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS)
      continue;

    /* Track the GPU with the most free VRAM — that's the one we'd use */
    nvmlMemory_t mem;
    if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
      long freeMB = (long)(mem.free >> 20);
      if (freeMB > *vramFreeMB) {
        *vramFreeMB = freeMB;
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS)
          *utilPct = (int)util.gpu;
      }
    }

    /* Track the highest compute capability across all GPUs */
    unsigned int maj = 0, min = 0;
    if (nvmlDeviceGetCudaComputeCapability(dev, &maj, &min) == NVML_SUCCESS) {
      int sm = (int)(maj * 10 + min);
      if (sm > *computeSM) *computeSM = sm;
    }
  }

  nvmlShutdown();
}
#endif /* USE_CUDA */


/****************************** PrintGpuInfo ***************************/
/* Prints GPU information to stdout at startup.                        */
void
PrintGpuInfo (void)
{
#ifdef USE_CUDA
  if (nvmlInit_v2() != NVML_SUCCESS) {
    printf ("  GPU          : CUDA build — NVML init failed (no driver?)\n");
    return;
  }

  unsigned int n = 0;
  nvmlDeviceGetCount_v2(&n);

  if (n == 0) {
    printf ("  GPU          : CUDA build — no devices found\n");
    nvmlShutdown();
    return;
  }

  for (unsigned int i = 0; i < n; i++) {
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS)
      continue;

    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = "(unknown)";
    nvmlDeviceGetName(dev, name, sizeof(name));

    nvmlMemory_t mem = {0};
    nvmlDeviceGetMemoryInfo(dev, &mem);

    unsigned int maj = 0, min = 0;
    nvmlDeviceGetCudaComputeCapability(dev, &maj, &min);

    printf ("  GPU %-2u       : %s  VRAM %lu MB  SM %u.%u\n",
            i, name, (unsigned long)(mem.total >> 20), maj, min);
  }

  nvmlShutdown();
#else
  printf ("  GPU          : none (CPU-only build)\n");
#endif
}


/*********************************** Coef ****************************/
int
Coef (PacketStruct * packet)
{
  int ccode = SUCCESS;
  char func[] = "Coef";

  ENTER (func);

  if (ccode == SUCCESS) {
    ccode = GetLoadAvg (&(packet->loadAvg));
    if (ccode != 0)
      packet->status = strdup("Error in GetLoadAvg");
  }

  if (ccode == SUCCESS) {
    ccode = GetMemFree (&(packet->memFree));
    if (ccode != 0)
      packet->status = strdup("Error in GetMemFree");
  }

  if (ccode == SUCCESS) {
    ccode = GetSwapFree (&(packet->swapFree));
    if (ccode != 0)
      packet->status = strdup("Error in GetSwapFree");
  }

  packet->cpuCount       = GetCpuCount();
  packet->activeNetworks = GetActiveNetworkCount();

#ifdef USE_CUDA
  GetGpuMetrics(&(packet->gpuCount),
                &(packet->gpuVramFreeMB),
                &(packet->gpuUtilPct),
                &(packet->gpuComputeSM));
#else
  packet->gpuCount    = 0;
  packet->gpuVramFreeMB = 0;
  packet->gpuUtilPct  = 0;
  packet->gpuComputeSM = 0;
#endif

  EXIT (func);
  return ccode;
}
