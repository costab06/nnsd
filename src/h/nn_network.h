#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nnsd.h"
#include "packet.h"

ThreadStruct *GetNetwork(int netID);
void RemoveNetwork(ThreadStruct *netPtr);
void DebugThreadList(char *str);
void ShutdownNetworks(void);
int AllocateWeightsAndVectors(ThreadStruct *netPtr);
int DeallocateNetwork(ThreadStruct *netPtr);
int InitializeNetworkFromPacket(PacketStruct *packet, ThreadStruct **netPtrAddr);
int InitializeNetworkFromMatrixFile(PacketStruct *packet, ThreadStruct **netPtrAddr);

#endif /* NN_NETWORK_H */
