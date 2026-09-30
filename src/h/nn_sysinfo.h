#ifndef NN_SYSINFO_H
#define NN_SYSINFO_H

#include "packet.h"

int  GetLoadAvg(float *load);
int  GetMemFree(long *free);
int  GetSwapFree(long *free);
int  Coef(PacketStruct *packet);
void PrintGpuInfo(void);

#endif /* NN_SYSINFO_H */
