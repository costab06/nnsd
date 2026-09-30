#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_harness.h"
#include "common.h"
#include "config.h"

/* Globals that would normally come from nnsd.c - we define them here */
int   runAsDaemon             = 0;
char *mountPoint              = NULL;
int   writeTrainingFailedFacts = 0;

/* debugPackets, debugMath, debugEnterExit come from log_utils.o */
extern int debugPackets;
extern int debugMath;
extern int debugEnterExit;

static void reset_globals(void)
{
    runAsDaemon             = 0;
    writeTrainingFailedFacts = 0;
    debugPackets             = 0;
    debugMath                = 0;
    debugEnterExit           = 0;

    if (mountPoint != NULL)
        free(mountPoint);
    mountPoint = calloc(1, ONE_KB);
    strcpy(mountPoint, "/opt/neural/data");
}

/* ------------------------------------------------------------------ */

static void test_config_missing_file(void)
{
    reset_globals();
    int ccode = ReadConfig("/tmp/nnsd_test_nonexistent_config_file_xyzzy.conf");
    ASSERT_EQ(ccode, FAILURE, "missing file should return FAILURE");
    ASSERT_EQ(runAsDaemon, 0, "runAsDaemon unchanged");
}

static void test_config_empty_file(void)
{
    reset_globals();
    const char *path = "/tmp/test_nnsd_empty.conf";
    FILE *fp = fopen(path, "w");
    if (fp) fclose(fp);

    int ccode = ReadConfig(path);
    ASSERT_EQ(ccode, SUCCESS, "empty file returns SUCCESS");
    remove(path);
}

static void test_config_all_keys(void)
{
    reset_globals();
    const char *path = "/tmp/test_nnsd_full.conf";
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "runAsDaemon = true\n");
        fprintf(fp, "mountPoint = /tmp/testmount\n");
        fprintf(fp, "writeTrainingFailedFacts = true\n");
        fprintf(fp, "debugPackets = true\n");
        fprintf(fp, "debugMath = true\n");
        fprintf(fp, "debugEnterExit = true\n");
        fclose(fp);
    }

    int ccode = ReadConfig(path);
    ASSERT_EQ(ccode, SUCCESS, "full config returns SUCCESS");
    ASSERT_EQ(runAsDaemon, 1, "runAsDaemon set to true");
    ASSERT_STR_EQ(mountPoint, "/tmp/testmount", "mountPoint set");
    ASSERT_EQ(writeTrainingFailedFacts, 1, "writeTrainingFailedFacts set");
    ASSERT_EQ(debugPackets, 1, "debugPackets set");
    ASSERT_EQ(debugMath, 1, "debugMath set");
    ASSERT_EQ(debugEnterExit, 1, "debugEnterExit set");
    remove(path);
}

static void test_config_false_values(void)
{
    reset_globals();
    /* pre-set all booleans to 1 */
    runAsDaemon              = 1;
    writeTrainingFailedFacts = 1;
    debugPackets             = 1;
    debugMath                = 1;
    debugEnterExit           = 1;

    const char *path = "/tmp/test_nnsd_false.conf";
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "runAsDaemon = false\n");
        fprintf(fp, "writeTrainingFailedFacts = false\n");
        fprintf(fp, "debugPackets = false\n");
        fprintf(fp, "debugMath = false\n");
        fprintf(fp, "debugEnterExit = false\n");
        fclose(fp);
    }

    int ccode = ReadConfig(path);
    ASSERT_EQ(ccode, SUCCESS, "false values config returns SUCCESS");
    ASSERT_EQ(runAsDaemon, 0, "runAsDaemon set to false");
    ASSERT_EQ(writeTrainingFailedFacts, 0, "writeTrainingFailedFacts set to false");
    ASSERT_EQ(debugPackets, 0, "debugPackets set to false");
    ASSERT_EQ(debugMath, 0, "debugMath set to false");
    ASSERT_EQ(debugEnterExit, 0, "debugEnterExit set to false");
    remove(path);
}

static void test_config_bool_numeric(void)
{
    reset_globals();
    const char *path = "/tmp/test_nnsd_numeric.conf";
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "runAsDaemon = 1\n");
        fprintf(fp, "writeTrainingFailedFacts = 0\n");
        fclose(fp);
    }

    ReadConfig(path);
    ASSERT_EQ(runAsDaemon, 1, "runAsDaemon = 1 (numeric)");
    ASSERT_EQ(writeTrainingFailedFacts, 0, "writeTrainingFailedFacts = 0 (numeric)");
    remove(path);
}

static void test_config_comments_and_blanks(void)
{
    reset_globals();
    const char *path = "/tmp/test_nnsd_comments.conf";
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "# This is a comment\n");
        fprintf(fp, "\n");
        fprintf(fp, "runAsDaemon = true\n");
        fprintf(fp, "\n");
        fprintf(fp, "# Another comment\n");
        fprintf(fp, "debugPackets = true\n");
        fclose(fp);
    }

    int ccode = ReadConfig(path);
    ASSERT_EQ(ccode, SUCCESS, "comments/blanks config returns SUCCESS");
    ASSERT_EQ(runAsDaemon, 1, "runAsDaemon parsed correctly past comments");
    ASSERT_EQ(debugPackets, 1, "debugPackets parsed correctly past comments");
    remove(path);
}

static void test_config_unknown_key(void)
{
    reset_globals();
    const char *path = "/tmp/test_nnsd_unknown.conf";
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "runAsDaemon = true\n");
        fprintf(fp, "unknownXYZ = value\n");
        fclose(fp);
    }

    int ccode = ReadConfig(path);
    ASSERT_EQ(ccode, SUCCESS, "unknown key config returns SUCCESS");
    ASSERT_EQ(runAsDaemon, 1, "valid key still parsed when unknown key present");
    remove(path);
}

static void test_config_malformed_line(void)
{
    reset_globals();
    const char *path = "/tmp/test_nnsd_malformed.conf";
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "badline\n");
        fclose(fp);
    }

    int ccode = ReadConfig(path);
    ASSERT_EQ(ccode, SUCCESS, "malformed line config returns SUCCESS");
    remove(path);
}

/* ------------------------------------------------------------------ */

int main(void)
{
    printf("=== test_config ===\n");
    RUN_TEST(test_config_missing_file);
    RUN_TEST(test_config_empty_file);
    RUN_TEST(test_config_all_keys);
    RUN_TEST(test_config_false_values);
    RUN_TEST(test_config_bool_numeric);
    RUN_TEST(test_config_comments_and_blanks);
    RUN_TEST(test_config_unknown_key);
    RUN_TEST(test_config_malformed_line);
    PRINT_RESULTS();

    if (mountPoint != NULL)
        free(mountPoint);

    return tests_failed > 0 ? 1 : 0;
}
