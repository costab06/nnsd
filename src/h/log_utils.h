#ifndef LOG_UTILS_H
#define LOG_UTILS_H

#include "packet.h"

/* function declarations */
void log_msg(const char *type, const char *msg);
void log_packet(const char *type, const PacketStruct *packet);
void log_debug1(const char *msg);
void log_debug2(const char *msg);
void close_log(void);

/* logging defines */
#define LOG_DEBUG_LEVEL1(x) if (debugLevel1 == TRUE) {log_debug1(x);}
#define LOG_DEBUG_LEVEL2(x) if (debugLevel2 == TRUE) {log_debug2(x);}

#define LOG_DEBUG_PACKET(x) if (debugPackets == TRUE) {log_packet("DEBUG_PACKETS:\t",x);}
#define LOG_DEBUG_MATH(x) if (debugMath == TRUE) {log_msg("DEBUG_MATH:\t",x);}
#define LOG_DEBUG_WEIGHTS(x) if (debugWeights == TRUE) {log_msg("DEBUG_WEIGHTS:\t",x);}
#define LOG_INFO(x) log_msg("INFO:\t",x)
#define LOG_WARN(x) log_msg("WARNING:\t",x)
#define LOG_ERROR(x) log_msg("ERROR:\t",x)
#define ENTER(x) if (debugEnterExit == TRUE) {log_msg("ENTER:\t",x);}
#define EXIT(x) if (debugEnterExit == TRUE) {log_msg("EXIT:\t",x);}

#endif /* LOG_UTILS_H */
