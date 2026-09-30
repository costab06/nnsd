#ifndef NN_PACKET_H
#define NN_PACKET_H

#include "packet.h"
#include "nnsd.h"

int ValidatePacket(PacketStruct *packet);
int ProcessPacketRPC(PacketStruct *packet);
PacketStruct *CreatePacketFromThreadStruct(ThreadStruct *netPtr, int packetType, int ccode, char *status);
int SendStatsPacket(ThreadStruct *netPtr);
int SendNetFinishedPacket(ThreadStruct *netPtr);

#endif /* NN_PACKET_H */
