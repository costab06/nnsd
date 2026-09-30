#include <stdio.h>
#include <stdlib.h>

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
int GetMemFree(long *free);
int GetSwapFree(long *free);
int GetLoadAvg(float *load);
int InitMutexes(void);
void DestroyMutexes(void);

/* ------------------------------------------------------------------ */

static void test_get_mem_free(void)
{
    long memFree = 0;
    int ccode = GetMemFree(&memFree);
    ASSERT_EQ(ccode, SUCCESS, "GetMemFree returns SUCCESS");
    ASSERT_TRUE(memFree > 0, "memFree > 0");
}

static void test_get_swap_free(void)
{
    long swapFree = 0;
    int ccode = GetSwapFree(&swapFree);
    ASSERT_EQ(ccode, SUCCESS, "GetSwapFree returns SUCCESS");
    ASSERT_TRUE(swapFree >= 0, "swapFree >= 0");
}

static void test_get_load_avg(void)
{
    float loadAvg = -1.0f;
    int ccode = GetLoadAvg(&loadAvg);
    ASSERT_EQ(ccode, SUCCESS, "GetLoadAvg returns SUCCESS");
    ASSERT_TRUE(loadAvg >= 0.0f, "loadAvg >= 0");
}

/* ------------------------------------------------------------------ */

int main(void)
{
    mountPoint = calloc(1, 1024);
    strcpy(mountPoint, "/tmp");

    InitMutexes();

    printf("=== test_sysinfo ===\n");
    RUN_TEST(test_get_mem_free);
    RUN_TEST(test_get_swap_free);
    RUN_TEST(test_get_load_avg);
    PRINT_RESULTS();

    DestroyMutexes();
    free(mountPoint);
    return tests_failed > 0 ? 1 : 0;
}
