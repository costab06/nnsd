#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "common.h"
#include "packet.h"
#include "log_utils.h"

extern int debugPackets;

/* ------------------------------------------------------------------ */

static PacketStruct *make_test_packet(void)
{
    PacketStruct *p = calloc(1, sizeof(PacketStruct));
    if (!p) return NULL;

    p->netID      = 42;
    p->packetType = COEF_PACKET;   /* 100 */
    p->ccode      = 0;
    p->status     = strdup("OK");
    p->src        = strdup("test");
    p->dest       = strdup("test");
    p->layers     = 2;
    p->neurons[0] = 2;
    p->neurons[1] = 1;
    p->learningRate              = 0.5f;
    p->momentum                  = 0.1f;
    p->trainingTolerance         = 0.05f;
    p->trainingSuccessPercentage = 90.0f;
    p->testingTolerance          = 0.1f;
    p->nudgeAmount               = 0.01f;
    p->mtxfile           = strdup("test");
    p->infile            = strdup("test");
    p->outfile           = strdup("test");
    p->stsfile           = strdup("test");
    p->failedfactsfile   = strdup("test");
    p->loadAvg           = 1.0f;
    p->memFree           = 1000000L;
    p->swapFree          = 500000L;
    p->inputFileHasHeader = 0;
    p->numInputRows      = 10;
    p->meanSumSquaredError  = 0.01f;
    p->percentFactsCorrect  = 95.0f;
    p->numberOfPasses       = 5;
    return p;
}

static void free_test_packet(PacketStruct *p)
{
    if (!p) return;
    free(p->status);
    free(p->src);
    free(p->dest);
    free(p->mtxfile);
    free(p->infile);
    free(p->outfile);
    free(p->stsfile);
    free(p->failedfactsfile);
    free(p);
}

/* ------------------------------------------------------------------ */

static void test_log_msg_info(void)
{
    log_msg("INFO:\t", "unit test info message\n");
    ASSERT_TRUE(1, "log_msg INFO did not crash");
}

static void test_log_msg_error(void)
{
    log_msg("ERROR:\t", "unit test error message\n");
    ASSERT_TRUE(1, "log_msg ERROR did not crash");
}

static void test_log_msg_empty(void)
{
    log_msg("INFO:\t", "");
    ASSERT_TRUE(1, "log_msg empty message did not crash");
}

static void test_log_packet_valid(void)
{
    PacketStruct *p = make_test_packet();
    log_packet("DEBUG_PACKETS:\t", p);
    ASSERT_TRUE(1, "log_packet valid did not crash");
    free_test_packet(p);
}

static void test_log_packet_null(void)
{
    log_packet("DEBUG_PACKETS:\t", NULL);
    ASSERT_TRUE(1, "log_packet NULL did not crash");
}

static void test_close_log(void)
{
    close_log();
    ASSERT_TRUE(1, "close_log did not crash");
}

/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== test_log ===\n");
    RUN_TEST(test_log_msg_info);
    RUN_TEST(test_log_msg_error);
    RUN_TEST(test_log_msg_empty);
    RUN_TEST(test_log_packet_valid);
    RUN_TEST(test_log_packet_null);
    RUN_TEST(test_close_log);
    PRINT_RESULTS();
    return tests_failed > 0 ? 1 : 0;
}
