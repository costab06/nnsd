#ifndef COMMON_H
#define COMMON_H

#include <pthread.h>

#define TRUE 1
#define FALSE 0

#define ONE_KB 1024
#define EIGHT_KB 8192

#define MAX_LAYERS 8

#define MODE_RUNNING 0
#define MODE_TRAINING 1
#define MODE_TESTING 2


#define NNSD_TCP_PORT  2122
#define NNCD_TCP_PORT  2123
#define NNSD_REST_PORT 8080

/* return values */

#define FAILURE -1
#define SUCCESS 0

#define FOPENERROR 1
#define FOPENERROR_TXT "ERROR: unable to open file\n"

#define FREADERROR 2
#define FREADERROR_TXT "ERROR: unable to read from file\n"

#define FCLOSEERROR 3
#define FCLOSEERROR_TXT "ERROR: unable to close file\n"

#define FILENOTEXISTERROR 4
#define FILENOTEXISTERROR_TXT "ERROR: file does not exist\n"

#define MALLOCERROR 5
#define MALLOCERROR_TXT "ERROR: unable to allocate memory\n" 

#define FWRITEERROR 6
#define FWRITEERROR_TXT "ERROR: unable to write to file\n"

#define PACKETTYPEERROR 7
#define PACKETTYPEERROR_TXT "ERROR: invalid packet type\n"

#define PACKETVALIDATIONERROR 8
#define PACKETVALIDATIONERROR_TXT "ERROR: packet failed validation\n"

#define NETWORKERROR 9
#define NETWORKERROR_TXT "ERROR: networking error\n"


#define CREATEMTXERROR  10
#define CREATEMTXERROR_TXT "ERROR: unable to create mutex\n"

#define DESTROYMTXERROR 11
#define DESTROYMTXERROR_TXT "ERROR: unable to destroy mutex\n"

#define FACTERROR 12
#define FACTERROR_TXT "ERROR: facts are in conflict\n"

#define NETWORKCANCELEDERROR 13
#define NETWORKCANCELEDERROR_TXT "ERROR: network cancelled error\n"

#define MATHERROR 14
#define MATHERROR_TXT "ERROR: math error\n"


#define NETNOTFOUNDERROR 15
#define NETNOTFOUNDERROR_TXT "ERROR: network not found\n"

#define NETALREADYPROCESSINGERROR 16
#define NETALREADYPROCESSINGERROR_TXT "ERROR: network already processing\n"

#define BEGINTHREADERROR 17
#define BEGINTHREADERROR_TXT "ERROR: unable to begin a thread\n"

#define MALFORMEDINPUTFILEERROR 18
#define MALFORMEDINPUTFILEERROR_TXT "ERROR: the input file is malformed\n"

#define RPCFUNCERROR 19
#define RPCFUNCERROR_TXT "ERROR: RPC function error\n"



/* activation function selectors */

#define ACTIVATION_SIGMOID    0
#define ACTIVATION_TANH       1
#define ACTIVATION_RELU       2
#define ACTIVATION_LEAKY_RELU 3

/* status defines */

#define STATUS_RUN 0
#define STATUS_SHUTDOWN 1

#endif /* COMMON_H */
