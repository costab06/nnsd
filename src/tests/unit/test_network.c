#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "test_harness.h"
#include "common.h"
#include "packet.h"
#include "log_utils.h"
#include "nnsd.h"

extern char *mountPoint;
extern int debugPackets;
extern int debugMath;
extern int debugEnterExit;

/* Function prototypes (defined in nnsd_test.o) */
int AllocateWeightsAndVectors(ThreadStruct *netPtr);
int DeallocateNetwork(ThreadStruct *netPtr);
int InitWtChange(ThreadStruct *netPtr);
int WriteMatrixFile(ThreadStruct *netPtr);
int InitializeNetworkFromMatrixFile(PacketStruct *packet, ThreadStruct **netPtrAddr);
int InitMutexes(void);
void DestroyMutexes(void);

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

    net->fileStructPtr->mtxfile         = strdup("/tmp/test_network.mtx");
    net->fileStructPtr->infile          = strdup("/tmp/test_network.csv");
    net->fileStructPtr->outfile         = strdup("/tmp/test_network_out.csv");
    net->fileStructPtr->stsfile         = strdup("/tmp/test_network.sts");
    net->fileStructPtr->failedfactsfile = strdup("/tmp/test_network_failed.csv");

    AllocateWeightsAndVectors(net);

    /* Set BIAS nodes */
    net->netStructPtr->vectors[0][in_n]  = 1.0f;
    net->netStructPtr->vectors[1][hid_n] = 1.0f;

    return net;
}

/* ------------------------------------------------------------------ */

static void test_allocate_weights_and_vectors(void)
{
    ThreadStruct *net = make_test_net(3, 4, 2);

    ASSERT_NOT_NULL(net->netStructPtr->vectors[0], "vectors[0] allocated");
    ASSERT_NOT_NULL(net->netStructPtr->vectors[1], "vectors[1] allocated");
    ASSERT_NOT_NULL(net->netStructPtr->vectors[2], "vectors[2] allocated");
    ASSERT_NOT_NULL(net->netStructPtr->weights[0], "weights[0] allocated");
    ASSERT_NOT_NULL(net->netStructPtr->weights[1], "weights[1] allocated");
    ASSERT_NOT_NULL(net->netStructPtr->weights[0][0], "weights[0][0] allocated");
    ASSERT_NOT_NULL(net->netStructPtr->minInputArray, "minInputArray allocated");
    ASSERT_NOT_NULL(net->netStructPtr->maxInputArray, "maxInputArray allocated");
    ASSERT_NOT_NULL(net->netStructPtr->expectedOutputArray, "expectedOutputArray allocated");

    DeallocateNetwork(net);
}

static void test_init_wt_change(void)
{
    /* 2-3-1 network */
    ThreadStruct *net = make_test_net(2, 3, 1);

    InitWtChange(net);

    /*
     * InitWtChange: for i in [0, layers-2]:
     *   for j in [0, neurons[i+1]):
     *     for k in [0, neurons[i]):
     *       previousWeightChange[i][j][k] = 0.1
     *
     * Layer 0: j < neurons[1]=3, k < neurons[0]=2
     * So [0][0][0] and [0][1][1] are set.
     */
    ASSERT_FLOAT_NEAR(net->netStructPtr->previousWeightChange[0][0][0], 0.1f, 0.001f,
                      "previousWeightChange[0][0][0] = 0.1");
    ASSERT_FLOAT_NEAR(net->netStructPtr->previousWeightChange[0][1][1], 0.1f, 0.001f,
                      "previousWeightChange[0][1][1] = 0.1");

    DeallocateNetwork(net);
}

static void test_deallocate_network(void)
{
    ThreadStruct *net = make_test_net(2, 2, 1);
    ASSERT_NOT_NULL(net, "net allocated");
    DeallocateNetwork(net);
    /* net is now freed by DeallocateNetwork */
    ASSERT_TRUE(1, "DeallocateNetwork did not crash");
}

static void test_write_matrix_file(void)
{
    ThreadStruct *net = make_test_net(2, 3, 1);

    net->netStructPtr->weights[0][0][0] = 0.123f;
    net->netStructPtr->weights[0][1][2] = 0.456f;
    net->netStructPtr->weights[1][0][0] = 0.789f;
    net->netStructPtr->minInputArray[0] = 0.0f;
    net->netStructPtr->maxInputArray[0] = 1.0f;

    free(net->fileStructPtr->mtxfile);
    net->fileStructPtr->mtxfile = strdup("/tmp/test_nnsd_unit.mtx");

    int ccode = WriteMatrixFile(net);
    ASSERT_EQ(ccode, SUCCESS, "WriteMatrixFile returns SUCCESS");

    FILE *fp = fopen("/tmp/test_nnsd_unit.mtx", "r");
    ASSERT_NOT_NULL(fp, "matrix file was created");
    if (fp) fclose(fp);

    DeallocateNetwork(net);
}

static void test_matrix_file_round_trip(void)
{
    /* Write a known network */
    ThreadStruct *net = make_test_net(2, 3, 1);

    net->netStructPtr->learningRate              = 0.25f;
    net->netStructPtr->momentum                  = 0.15f;
    net->netStructPtr->trainingTolerance         = 0.05f;
    net->netStructPtr->trainingSuccessPercentage = 85.0f;
    net->netStructPtr->testingTolerance          = 0.08f;
    net->netStructPtr->weights[0][0][0] = 0.111f;
    net->netStructPtr->weights[0][0][1] = 0.222f;
    net->netStructPtr->weights[1][0][0] = 0.333f;
    net->netStructPtr->minInputArray[0] = -1.0f;
    net->netStructPtr->maxInputArray[0] =  1.0f;
    net->netStructPtr->minInputArray[1] =  0.0f;
    net->netStructPtr->maxInputArray[1] = 10.0f;
    net->netStructPtr->minOutputArray[0] = 0.0f;
    net->netStructPtr->maxOutputArray[0] = 1.0f;

    free(net->fileStructPtr->mtxfile);
    net->fileStructPtr->mtxfile = strdup("/tmp/test_nnsd_roundtrip.mtx");

    int wcode = WriteMatrixFile(net);
    ASSERT_EQ(wcode, SUCCESS, "WriteMatrixFile round-trip write");

    /* Read it back via InitializeNetworkFromMatrixFile */
    PacketStruct packet;
    memset(&packet, 0, sizeof(packet));
    packet.netID           = 42;
    packet.packetType      = TRAIN_PACKET;
    packet.src             = "roundtrip_test";
    packet.nudgeAmount     = 0.01f;
    packet.inputFileHasHeader = FALSE;
    packet.mtxfile         = "/tmp/test_nnsd_roundtrip.mtx";
    /* infile must exist for LoadAndValidateInputFile – create a minimal one */
    packet.infile          = "/tmp/test_nnsd_roundtrip_in.csv";
    packet.outfile         = "/tmp/test_nnsd_roundtrip_out.csv";
    packet.stsfile         = "/tmp/test_nnsd_roundtrip.sts";
    packet.failedfactsfile = "/tmp/test_nnsd_roundtrip_failed.csv";

    /* Create a minimal input CSV (2 inputs + 1 output for a 2-3-1 net) */
    FILE *csv = fopen("/tmp/test_nnsd_roundtrip_in.csv", "w");
    if (csv) {
        fprintf(csv, "0.1,0.2,0.9\n");
        fprintf(csv, "0.8,0.7,0.1\n");
        fclose(csv);
    }

    ThreadStruct *net2 = NULL;
    int rcode = InitializeNetworkFromMatrixFile(&packet, &net2);
    ASSERT_EQ(rcode, SUCCESS, "InitializeNetworkFromMatrixFile returns SUCCESS");
    ASSERT_NOT_NULL(net2, "net2 not null after read");

    if (net2 != NULL && net2->netStructPtr != NULL) {
        ASSERT_EQ(net2->netStructPtr->layers, net->netStructPtr->layers, "layers round-trip");
        ASSERT_EQ(net2->netStructPtr->neurons[0], net->netStructPtr->neurons[0], "neurons[0] round-trip");
        ASSERT_EQ(net2->netStructPtr->neurons[1], net->netStructPtr->neurons[1], "neurons[1] round-trip");
        ASSERT_EQ(net2->netStructPtr->neurons[2], net->netStructPtr->neurons[2], "neurons[2] round-trip");
        ASSERT_FLOAT_NEAR(net2->netStructPtr->learningRate, net->netStructPtr->learningRate, 0.001f,
                          "learningRate round-trip");
        ASSERT_FLOAT_NEAR(net2->netStructPtr->momentum, net->netStructPtr->momentum, 0.001f,
                          "momentum round-trip");
        ASSERT_FLOAT_NEAR(net2->netStructPtr->weights[0][0][0], net->netStructPtr->weights[0][0][0], 0.001f,
                          "weight[0][0][0] round-trip");
        ASSERT_FLOAT_NEAR(net2->netStructPtr->weights[1][0][0], net->netStructPtr->weights[1][0][0], 0.001f,
                          "weight[1][0][0] round-trip");
        ASSERT_FLOAT_NEAR(net2->netStructPtr->minInputArray[0], net->netStructPtr->minInputArray[0], 0.001f,
                          "minInputArray[0] round-trip");
        ASSERT_FLOAT_NEAR(net2->netStructPtr->maxInputArray[0], net->netStructPtr->maxInputArray[0], 0.001f,
                          "maxInputArray[0] round-trip");
        DeallocateNetwork(net2);
    }

    DeallocateNetwork(net);
    remove("/tmp/test_nnsd_roundtrip.mtx");
    remove("/tmp/test_nnsd_roundtrip_in.csv");
    remove("/tmp/test_nnsd_roundtrip_out.csv");
    remove("/tmp/test_nnsd_roundtrip.sts");
    remove("/tmp/test_nnsd_roundtrip_failed.csv");
}

/* ------------------------------------------------------------------ */

int main(void)
{
    mountPoint = calloc(1, 1024);
    strcpy(mountPoint, "/tmp");

    InitMutexes();

    printf("=== test_network ===\n");
    RUN_TEST(test_allocate_weights_and_vectors);
    RUN_TEST(test_init_wt_change);
    RUN_TEST(test_deallocate_network);
    RUN_TEST(test_write_matrix_file);
    RUN_TEST(test_matrix_file_round_trip);
    PRINT_RESULTS();

    DestroyMutexes();
    free(mountPoint);
    return tests_failed > 0 ? 1 : 0;
}
