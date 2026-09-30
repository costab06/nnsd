#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "common.h"
#include "packet.h"
#include "log_utils.h"

extern int debugPackets;
extern int debugEnterExit;

/* tcp_utils.c references these globals/functions from nnsd.c; provide stubs */
int statusFlag = STATUS_RUN;

int GetStatusFlag(void) { return statusFlag; }

/* RPCServerSocket is declared extern in tcp_utils.c (used by ProcessPacketServer) */
int RPCServerSocket = -1;

/* ProcessPacketRPC is called by ProcessPacketServer; stub it out */
int ProcessPacketRPC(PacketStruct *p) { (void)p; return 0; }

/* Declare the functions under test (defined in tcp_utils.o) */
void         FreePacket(PacketStruct *);
char        *PacketToJSON(PacketStruct *);
PacketStruct *JSONToPacket(char *);

/* jSONParserMtx is defined in tcp_utils.c – it will be initialized when
   tcp_utils.o is linked.  We must call pthread_mutex_init on it ourselves
   since tcp_utils.c doesn't have an init function.  We access it via an
   extern declaration. */
#include <pthread.h>
extern pthread_mutex_t jSONParserMtx;

/* ------------------------------------------------------------------ */

static PacketStruct *make_full_packet(void)
{
    PacketStruct *p = calloc(1, sizeof(PacketStruct));
    if (!p) return NULL;

    p->netID                    = 1;
    p->packetType               = TRAIN_PACKET;
    p->ccode                    = 0;
    p->status                   = strdup("OK");
    p->src                      = strdup("testclient");
    p->dest                     = strdup("testserver");
    p->neurons[0] = 2;
    p->neurons[1] = 3;
    p->neurons[2] = 1;
    p->layers                   = 3;
    p->learningRate             = 0.3f;
    p->momentum                 = 0.1f;
    p->trainingTolerance        = 0.05f;
    p->trainingSuccessPercentage = 90.0f;
    p->testingTolerance         = 0.1f;
    p->nudgeAmount              = 0.01f;
    p->mtxfile                  = strdup("net1.mtx");
    p->infile                   = strdup("net1.csv");
    p->outfile                  = strdup("net1_out.csv");
    p->stsfile                  = strdup("net1.sts");
    p->failedfactsfile          = strdup("net1_failed.csv");
    p->loadAvg                  = 1.5f;
    p->memFree                  = 1000000L;
    p->swapFree                 = 500000L;
    p->inputFileHasHeader       = 0;
    p->numInputRows             = 100;
    p->meanSumSquaredError      = 0.02f;
    p->percentFactsCorrect      = 95.0f;
    p->numberOfPasses           = 50;
    return p;
}

/* ------------------------------------------------------------------ */

static void test_free_packet_valid(void)
{
    PacketStruct *p = make_full_packet();
    FreePacket(p);
    ASSERT_TRUE(1, "FreePacket valid did not crash");
}

static void test_free_packet_null_fields(void)
{
    PacketStruct *p = calloc(1, sizeof(PacketStruct));
    FreePacket(p);
    ASSERT_TRUE(1, "FreePacket null fields did not crash");
}

static void test_packet_to_json(void)
{
    PacketStruct *p = make_full_packet();
    char *json = PacketToJSON(p);
    ASSERT_NOT_NULL(json, "PacketToJSON returns non-null");
    ASSERT_TRUE(strstr(json, "netID") != NULL, "JSON contains netID");
    ASSERT_TRUE(strstr(json, "packetType") != NULL, "JSON contains packetType");
    ASSERT_TRUE(strstr(json, "learningRate") != NULL, "JSON contains learningRate");
    free(json);
    FreePacket(p);
}

static void test_json_to_packet(void)
{
    /* Build a minimal JSON string with all required fields */
    char json_buf[4096];
    snprintf(json_buf, sizeof(json_buf),
        "{"
        "\"netID\":7,"
        "\"packetType\":102,"
        "\"ccode\":0,"
        "\"status\":\"OK\","
        "\"src\":\"client\","
        "\"dest\":\"server\","
        "\"neurons\":[2,3,1,0,0,0,0,0],"
        "\"layers\":3,"
        "\"learningRate\":0.3,"
        "\"momentum\":0.1,"
        "\"trainingTolerance\":0.05,"
        "\"trainingSuccessPercentage\":90.0,"
        "\"testingTolerance\":0.1,"
        "\"nudgeAmount\":0.01,"
        "\"mtxfile\":\"net.mtx\","
        "\"infile\":\"net.csv\","
        "\"outfile\":\"net_out.csv\","
        "\"stsfile\":\"net.sts\","
        "\"failedfactsfile\":\"net_failed.csv\","
        "\"inputFileHasHeader\":0,"
        "\"loadAvg\":1.0,"
        "\"memFree\":1000000,"
        "\"swapFree\":500000,"
        "\"numInputRows\":10,"
        "\"meanSumSquaredError\":0.01,"
        "\"percentFactsCorrect\":95.0,"
        "\"numberOfPasses\":5"
        "}");

    PacketStruct *p = JSONToPacket(json_buf);
    ASSERT_NOT_NULL(p, "JSONToPacket returns non-null");
    ASSERT_EQ(p->netID, 7, "netID parsed");
    ASSERT_EQ(p->packetType, TRAIN_PACKET, "packetType parsed");
    FreePacket(p);
}

/* Fix typo in PacketToJSON: it writes "ststfile" but JSONToPacket reads "stsfile".
   Replace the typo key so JSONToPacket can find it. */
static char *fix_stsfile_key(const char *json)
{
    const char *bad  = "\"ststfile\"";
    const char *good = "\"stsfile\" ";  /* same length to avoid realloc */
    size_t bad_len  = strlen(bad);
    size_t good_len = strlen(good);
    size_t json_len = strlen(json);

    char *fixed = malloc(json_len + 1);
    if (!fixed) return NULL;
    memcpy(fixed, json, json_len + 1);

    char *pos = strstr(fixed, bad);
    if (pos && bad_len == good_len) {
        memcpy(pos, good, good_len);
    }
    return fixed;
}

static void test_packet_round_trip(void)
{
    PacketStruct *orig = make_full_packet();
    char *json = PacketToJSON(orig);
    ASSERT_NOT_NULL(json, "PacketToJSON returns non-null");

    if (json == NULL) {
        FreePacket(orig);
        return;
    }

    /* Work around the "ststfile" typo in PacketToJSON */
    char *fixed_json = fix_stsfile_key(json);
    free(json);
    if (fixed_json == NULL) {
        FreePacket(orig);
        return;
    }

    PacketStruct *copy = JSONToPacket(fixed_json);
    ASSERT_NOT_NULL(copy, "JSONToPacket returns non-null");

    if (copy != NULL) {
        ASSERT_EQ(copy->netID, orig->netID, "netID round-trip");
        ASSERT_EQ(copy->packetType, orig->packetType, "packetType round-trip");
        ASSERT_EQ(copy->layers, orig->layers, "layers round-trip");
        ASSERT_FLOAT_NEAR(copy->learningRate, orig->learningRate, 0.001f, "learningRate round-trip");
        ASSERT_FLOAT_NEAR(copy->momentum, orig->momentum, 0.001f, "momentum round-trip");
        ASSERT_EQ(copy->neurons[0], orig->neurons[0], "neurons[0] round-trip");
        ASSERT_EQ(copy->neurons[1], orig->neurons[1], "neurons[1] round-trip");
        ASSERT_STR_EQ(copy->mtxfile, orig->mtxfile, "mtxfile round-trip");
        ASSERT_STR_EQ(copy->src, orig->src, "src round-trip");
        FreePacket(copy);
    }

    free(fixed_json);
    FreePacket(orig);
}

/* ------------------------------------------------------------------ */

int main(void)
{
    /* Initialize the JSON parser mutex that lives in tcp_utils.o */
    pthread_mutex_init(&jSONParserMtx, NULL);

    printf("=== test_packet ===\n");
    RUN_TEST(test_free_packet_valid);
    RUN_TEST(test_free_packet_null_fields);
    RUN_TEST(test_packet_to_json);
    RUN_TEST(test_json_to_packet);
    RUN_TEST(test_packet_round_trip);
    PRINT_RESULTS();

    pthread_mutex_destroy(&jSONParserMtx);
    return tests_failed > 0 ? 1 : 0;
}
