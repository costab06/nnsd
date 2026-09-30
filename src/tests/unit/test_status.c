#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "common.h"
#include "packet.h"
#include "log_utils.h"
#include "nnsd.h"

extern char *mountPoint;
extern int statusFlag;
extern int debugPackets;
extern int debugMath;
extern int debugEnterExit;

/* Function prototypes (defined in nnsd_test.o) */
int SetStatusFlag(int status);
int GetStatusFlag(void);
int SetNetStatusFlag(ThreadStruct *netPtr, int status);
int GetNetStatusFlag(ThreadStruct *netPtr);
int ValidatePacket(PacketStruct *packet);
int InitMutexes(void);
void DestroyMutexes(void);
int AllocateWeightsAndVectors(ThreadStruct *netPtr);
int DeallocateNetwork(ThreadStruct *netPtr);

/* ------------------------------------------------------------------ */

static ThreadStruct *make_test_net(int in_n, int hid_n, int out_n)
{
    ThreadStruct *net = calloc(1, sizeof(ThreadStruct));
    if (!net) return NULL;

    net->netStructPtr = calloc(1, sizeof(NetStruct));
    if (!net->netStructPtr) { free(net); return NULL; }

    net->fileStructPtr = calloc(1, sizeof(FileStruct));
    if (!net->fileStructPtr) { free(net->netStructPtr); free(net); return NULL; }

    net->netID       = 1;
    net->networkType = MODE_TRAINING;
    net->src         = strdup("test");
    net->statusFlag  = STATUS_RUN;
    net->nudgeAmount = 0.1f;
    net->flink       = NULL;

    pthread_mutex_init(&net->threadInfoHoldMtx, NULL);

    net->netStructPtr->layers     = 3;
    net->netStructPtr->neurons[0] = in_n;
    net->netStructPtr->neurons[1] = hid_n;
    net->netStructPtr->neurons[2] = out_n;

    net->netStructPtr->learningRate              = 0.3f;
    net->netStructPtr->momentum                  = 0.1f;
    net->netStructPtr->trainingTolerance         = 0.1f;
    net->netStructPtr->trainingSuccessPercentage = 90.0f;
    net->netStructPtr->testingTolerance          = 0.1f;

    net->fileStructPtr->mtxfile         = strdup("/tmp/test_status.mtx");
    net->fileStructPtr->infile          = strdup("/tmp/test_status.csv");
    net->fileStructPtr->outfile         = strdup("/tmp/test_status_out.csv");
    net->fileStructPtr->stsfile         = strdup("/tmp/test_status.sts");
    net->fileStructPtr->failedfactsfile = strdup("/tmp/test_status_failed.csv");

    AllocateWeightsAndVectors(net);

    net->netStructPtr->vectors[0][in_n]  = 1.0f;
    net->netStructPtr->vectors[1][hid_n] = 1.0f;

    return net;
}

/* ------------------------------------------------------------------ */

static void test_set_get_status_flag(void)
{
    SetStatusFlag(STATUS_RUN);
    ASSERT_EQ(GetStatusFlag(), STATUS_RUN, "status flag is STATUS_RUN");

    SetStatusFlag(STATUS_SHUTDOWN);
    ASSERT_EQ(GetStatusFlag(), STATUS_SHUTDOWN, "status flag is STATUS_SHUTDOWN");

    /* reset */
    SetStatusFlag(STATUS_RUN);
}

static void test_set_get_net_status_flag(void)
{
    ThreadStruct *net = make_test_net(2, 2, 1);

    SetNetStatusFlag(net, STATUS_SHUTDOWN);
    ASSERT_EQ(GetNetStatusFlag(net), STATUS_SHUTDOWN, "net status is STATUS_SHUTDOWN");

    SetNetStatusFlag(net, STATUS_RUN);
    ASSERT_EQ(GetNetStatusFlag(net), STATUS_RUN, "net status is STATUS_RUN");

    DeallocateNetwork(net);
}

static void test_validate_packet_coef(void)
{
    PacketStruct packet;
    memset(&packet, 0, sizeof(packet));
    packet.packetType = COEF_PACKET;
    ASSERT_EQ(ValidatePacket(&packet), SUCCESS, "COEF_PACKET is valid");
}

static void test_validate_packet_shutdown(void)
{
    PacketStruct packet;
    memset(&packet, 0, sizeof(packet));
    packet.packetType = SHUTDOWN_PACKET;
    ASSERT_EQ(ValidatePacket(&packet), SUCCESS, "SHUTDOWN_PACKET is valid");
}

static void test_validate_packet_type_too_low(void)
{
    PacketStruct packet;
    memset(&packet, 0, sizeof(packet));
    packet.packetType = 50;
    ASSERT_NE(ValidatePacket(&packet), SUCCESS, "packet type too low is invalid");
}

static void test_validate_packet_type_too_high(void)
{
    PacketStruct packet;
    memset(&packet, 0, sizeof(packet));
    packet.packetType = 200;
    ASSERT_NE(ValidatePacket(&packet), SUCCESS, "packet type too high is invalid");
}

static void test_validate_packet_run_missing_file(void)
{
    PacketStruct packet;
    memset(&packet, 0, sizeof(packet));
    packet.packetType = RUN_PACKET;
    packet.infile     = "/tmp/nonexistent_test_file_xyzzy_abc123.csv";
    ASSERT_NE(ValidatePacket(&packet), SUCCESS, "RUN_PACKET with missing file is invalid");
}

static void test_validate_packet_run_existing_file(void)
{
    FILE *fp = fopen("/tmp/test_validate_input.csv", "w");
    if (fp) { fprintf(fp, "1,0\n"); fclose(fp); }

    PacketStruct packet;
    memset(&packet, 0, sizeof(packet));
    packet.packetType = RUN_PACKET;
    packet.infile     = "/tmp/test_validate_input.csv";
    ASSERT_EQ(ValidatePacket(&packet), SUCCESS, "RUN_PACKET with existing file is valid");

    remove("/tmp/test_validate_input.csv");
}

/* ------------------------------------------------------------------ */

int main(void)
{
    mountPoint = calloc(1, 1024);
    strcpy(mountPoint, "/tmp");

    InitMutexes();

    printf("=== test_status ===\n");
    RUN_TEST(test_set_get_status_flag);
    RUN_TEST(test_set_get_net_status_flag);
    RUN_TEST(test_validate_packet_coef);
    RUN_TEST(test_validate_packet_shutdown);
    RUN_TEST(test_validate_packet_type_too_low);
    RUN_TEST(test_validate_packet_type_too_high);
    RUN_TEST(test_validate_packet_run_missing_file);
    RUN_TEST(test_validate_packet_run_existing_file);
    PRINT_RESULTS();

    DestroyMutexes();
    free(mountPoint);
    return tests_failed > 0 ? 1 : 0;
}
