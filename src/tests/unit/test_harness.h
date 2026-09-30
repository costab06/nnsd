#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdio.h>
#include <math.h>
#include <string.h>

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_TRUE(expr, msg) do { \
    tests_run++; \
    if (expr) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
    } \
} while(0)

#define ASSERT_FALSE(expr, msg) do { \
    tests_run++; \
    if (!(expr)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    tests_run++; \
    if ((a) == (b)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
    } \
} while(0)

#define ASSERT_NE(a, b, msg) do { \
    tests_run++; \
    if ((a) != (b)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
    } \
} while(0)

#define ASSERT_NULL(ptr, msg) do { \
    tests_run++; \
    if ((ptr) == NULL) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
    } \
} while(0)

#define ASSERT_NOT_NULL(ptr, msg) do { \
    tests_run++; \
    if ((ptr) != NULL) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
    } \
} while(0)

#define ASSERT_FLOAT_NEAR(a, b, eps, msg) do { \
    tests_run++; \
    if (fabs((double)(a) - (double)(b)) <= (double)(eps)) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
    } \
} while(0)

#define ASSERT_STR_EQ(a, b, msg) do { \
    tests_run++; \
    if ((a) != NULL && (b) != NULL && strcmp((a), (b)) == 0) { \
        tests_passed++; \
    } else { \
        tests_failed++; \
        fprintf(stderr, "  FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
    } \
} while(0)

#define RUN_TEST(fn) do { \
    printf("  Running " #fn "...\n"); \
    fn(); \
} while(0)

#define PRINT_RESULTS() do { \
    printf("=== Results: %d passed, %d failed, %d total ===\n", \
           tests_passed, tests_failed, tests_run); \
} while(0)

#endif /* TEST_HARNESS_H */
