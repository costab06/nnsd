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
int SigmoidTransferFunction(float *v);
int ScaleDownFact(float *v, int length, float *min, float *max);
int ScaleUpFact(float *v, int length, float *min, float *max);
int TestFact(ThreadStruct *netPtr, float *sumSquaredError);
int FeedForward(ThreadStruct *netPtr);
int BackProp(ThreadStruct *netPtr);
int AllocateWeightsAndVectors(ThreadStruct *netPtr);
int DeallocateNetwork(ThreadStruct *netPtr);
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

    net->fileStructPtr->mtxfile         = strdup("/tmp/test_math.mtx");
    net->fileStructPtr->infile          = strdup("/tmp/test_math.csv");
    net->fileStructPtr->outfile         = strdup("/tmp/test_math_out.csv");
    net->fileStructPtr->stsfile         = strdup("/tmp/test_math.sts");
    net->fileStructPtr->failedfactsfile = strdup("/tmp/test_math_failed.csv");

    AllocateWeightsAndVectors(net);

    /* Set BIAS nodes */
    net->netStructPtr->vectors[0][in_n]  = 1.0f;
    net->netStructPtr->vectors[1][hid_n] = 1.0f;

    return net;
}

/* ------------------------------------------------------------------ */

static void test_sigmoid_zero(void)
{
    float v = 0.0f;
    SigmoidTransferFunction(&v);
    ASSERT_FLOAT_NEAR(v, 0.5f, 0.001f, "sigmoid(0) = 0.5");
}

static void test_sigmoid_large_positive(void)
{
    float v = 100.0f;
    SigmoidTransferFunction(&v);
    ASSERT_TRUE(v > 0.99f, "sigmoid(100) near 1.0");
}

static void test_sigmoid_large_negative(void)
{
    float v = -100.0f;
    SigmoidTransferFunction(&v);
    ASSERT_TRUE(v < 0.01f, "sigmoid(-100) near 0.0");
}

static void test_sigmoid_positive(void)
{
    float v = 2.0f;
    SigmoidTransferFunction(&v);
    ASSERT_TRUE(v > 0.5f && v < 1.0f, "sigmoid(2) in (0.5, 1.0)");
}

static void test_sigmoid_negative(void)
{
    float v = -2.0f;
    SigmoidTransferFunction(&v);
    ASSERT_TRUE(v > 0.0f && v < 0.5f, "sigmoid(-2) in (0.0, 0.5)");
}

static void test_scale_down_at_min(void)
{
    float v[1]   = { 0.0f };
    float min[1] = { 0.0f };
    float max[1] = { 10.0f };
    ScaleDownFact(v, 1, min, max);
    ASSERT_FLOAT_NEAR(v[0], 0.0f, 0.001f, "scale_down min -> 0.0");
}

static void test_scale_down_at_max(void)
{
    float v[1]   = { 10.0f };
    float min[1] = { 0.0f };
    float max[1] = { 10.0f };
    ScaleDownFact(v, 1, min, max);
    ASSERT_FLOAT_NEAR(v[0], 1.0f, 0.001f, "scale_down max -> 1.0");
}

static void test_scale_down_midpoint(void)
{
    float v[1]   = { 5.0f };
    float min[1] = { 0.0f };
    float max[1] = { 10.0f };
    ScaleDownFact(v, 1, min, max);
    ASSERT_FLOAT_NEAR(v[0], 0.5f, 0.001f, "scale_down midpoint -> 0.5");
}

static void test_scale_down_multi(void)
{
    float v[2]   = { 2.0f, 8.0f };
    float min[2] = { 0.0f, 0.0f };
    float max[2] = { 10.0f, 10.0f };
    ScaleDownFact(v, 2, min, max);
    ASSERT_FLOAT_NEAR(v[0], 0.2f, 0.001f, "scale_down multi[0]");
    ASSERT_FLOAT_NEAR(v[1], 0.8f, 0.001f, "scale_down multi[1]");
}

static void test_scale_up_at_zero(void)
{
    float v[1]   = { 0.0f };
    float min[1] = { 0.0f };
    float max[1] = { 10.0f };
    ScaleUpFact(v, 1, min, max);
    ASSERT_FLOAT_NEAR(v[0], 0.0f, 0.001f, "scale_up 0.0 -> min");
}

static void test_scale_up_at_one(void)
{
    float v[1]   = { 1.0f };
    float min[1] = { 0.0f };
    float max[1] = { 10.0f };
    ScaleUpFact(v, 1, min, max);
    ASSERT_FLOAT_NEAR(v[0], 10.0f, 0.001f, "scale_up 1.0 -> max");
}

static void test_scale_round_trip(void)
{
    float orig[2] = { 3.0f, 7.0f };
    float min[2]  = { 0.0f, 0.0f };
    float max[2]  = { 10.0f, 10.0f };
    float v[2]    = { orig[0], orig[1] };
    ScaleDownFact(v, 2, min, max);
    ScaleUpFact(v, 2, min, max);
    ASSERT_FLOAT_NEAR(v[0], orig[0], 0.001f, "scale round-trip [0]");
    ASSERT_FLOAT_NEAR(v[1], orig[1], 0.001f, "scale round-trip [1]");
}

static void test_feedforward_output_in_range(void)
{
    ThreadStruct *net = make_test_net(2, 2, 1);
    /* set all weights to 0.5 */
    int i, j, k;
    for (i = 0; i < net->netStructPtr->layers - 1; i++)
        for (j = 0; j < net->netStructPtr->neurons[i] + 1; j++)
            for (k = 0; k < net->netStructPtr->neurons[i + 1]; k++)
                net->netStructPtr->weights[i][j][k] = 0.5f;

    net->netStructPtr->vectors[0][0] = 0.8f;
    net->netStructPtr->vectors[0][1] = 0.3f;
    net->netStructPtr->vectors[0][2] = 1.0f; /* bias */

    FeedForward(net);

    float out = net->netStructPtr->vectors[2][0];
    ASSERT_TRUE(out > 0.0f && out < 1.0f, "FeedForward output in (0,1)");

    DeallocateNetwork(net);
    /* DeallocateNetwork already frees net itself */
}

static void test_testfact_correct(void)
{
    ThreadStruct *net = make_test_net(2, 2, 1);
    net->networkType = MODE_TRAINING;
    net->netStructPtr->vectors[2][0]          = 0.8f;
    net->netStructPtr->expectedOutputArray[0] = 0.82f; /* diff=0.02 < tol=0.1 */

    float sse = 0.0f;
    int result = TestFact(net, &sse);
    ASSERT_EQ(result, TRUE, "TestFact correct within tolerance");
    ASSERT_TRUE(sse >= 0.0f, "SSE non-negative");

    DeallocateNetwork(net);
}

static void test_testfact_incorrect(void)
{
    ThreadStruct *net = make_test_net(2, 2, 1);
    net->networkType = MODE_TRAINING;
    net->netStructPtr->vectors[2][0]          = 0.2f;
    net->netStructPtr->expectedOutputArray[0] = 0.9f; /* diff=0.7 > tol=0.1 */

    float sse = 0.0f;
    int result = TestFact(net, &sse);
    ASSERT_EQ(result, FALSE, "TestFact incorrect outside tolerance");

    DeallocateNetwork(net);
}

static void test_testfact_testing_mode(void)
{
    ThreadStruct *net = make_test_net(2, 2, 1);
    net->networkType = MODE_TESTING;
    net->netStructPtr->vectors[2][0]          = 0.8f;
    net->netStructPtr->expectedOutputArray[0] = 0.82f; /* diff=0.02 < testingTol=0.1 */

    float sse = 0.0f;
    int result = TestFact(net, &sse);
    ASSERT_EQ(result, TRUE, "TestFact testing mode correct within tolerance");

    DeallocateNetwork(net);
}

static void test_backprop_updates_weights(void)
{
    ThreadStruct *net = make_test_net(2, 2, 1);
    int i, j, k;

    /* set weights to 0.5 and previousWeightChange to 0 */
    for (i = 0; i < net->netStructPtr->layers - 1; i++)
        for (j = 0; j < net->netStructPtr->neurons[i] + 1; j++)
            for (k = 0; k < net->netStructPtr->neurons[i + 1]; k++) {
                net->netStructPtr->weights[i][j][k]              = 0.5f;
                net->netStructPtr->previousWeightChange[i][j][k] = 0.0f;
            }

    net->netStructPtr->vectors[0][0] = 0.9f;
    net->netStructPtr->vectors[0][1] = 0.1f;
    net->netStructPtr->vectors[0][2] = 1.0f; /* bias */

    FeedForward(net);
    net->netStructPtr->expectedOutputArray[0] = 0.9f;

    float w_before = net->netStructPtr->weights[1][0][0];
    BackProp(net);
    float w_after = net->netStructPtr->weights[1][0][0];

    ASSERT_TRUE(w_before != w_after, "BackProp changed weights");

    DeallocateNetwork(net);
}

/* ------------------------------------------------------------------ */

int main(void)
{
    mountPoint = calloc(1, 1024);
    strcpy(mountPoint, "/tmp");

    InitMutexes();

    printf("=== test_math ===\n");
    RUN_TEST(test_sigmoid_zero);
    RUN_TEST(test_sigmoid_large_positive);
    RUN_TEST(test_sigmoid_large_negative);
    RUN_TEST(test_sigmoid_positive);
    RUN_TEST(test_sigmoid_negative);
    RUN_TEST(test_scale_down_at_min);
    RUN_TEST(test_scale_down_at_max);
    RUN_TEST(test_scale_down_midpoint);
    RUN_TEST(test_scale_down_multi);
    RUN_TEST(test_scale_up_at_zero);
    RUN_TEST(test_scale_up_at_one);
    RUN_TEST(test_scale_round_trip);
    RUN_TEST(test_feedforward_output_in_range);
    RUN_TEST(test_testfact_correct);
    RUN_TEST(test_testfact_incorrect);
    RUN_TEST(test_testfact_testing_mode);
    RUN_TEST(test_backprop_updates_weights);
    PRINT_RESULTS();

    DestroyMutexes();
    free(mountPoint);
    return tests_failed > 0 ? 1 : 0;
}
