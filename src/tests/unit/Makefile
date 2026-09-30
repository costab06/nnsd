CC      = gcc
CFLAGS  = -g -O0 -D_REENTRANT -Wall -Wno-unused-variable
INCLUDES = -I../../h -I../../lib/cJSON -I../../lib/libcsv
BUILDDIR = ../../../cmake-build-debug

# Pre-built object files (built by the top-level build)
LOGOBJ   = $(BUILDDIR)/log_utils.o
CFGOBJ   = $(BUILDDIR)/config.o
TCPOBJ   = $(BUILDDIR)/tcp_utils.o
CJSONOBJ = $(BUILDDIR)/cJSON.o
CSVOBJ   = $(BUILDDIR)/libcsv.o
DAEMONOBJ = $(BUILDDIR)/daemon.o

# nnsd.c compiled with main renamed so tests can provide their own main
NNSDOBJ = $(BUILDDIR)/nnsd_test.o

$(NNSDOBJ): ../../nnsd/nnsd.c
	$(CC) $(CFLAGS) $(INCLUDES) -Dmain=nnsd_main -c $< -o $@

# Objects used by tests that link nnsd
NNSD_DEPS = $(NNSDOBJ) $(LOGOBJ) $(TCPOBJ) $(CJSONOBJ) $(CSVOBJ) $(CFGOBJ) $(DAEMONOBJ)

# ---- test_config ----
$(BUILDDIR)/test_config: test_config.c $(LOGOBJ) $(CFGOBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lpthread -lm

# ---- test_log ----
$(BUILDDIR)/test_log: test_log.c $(LOGOBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lpthread -lm

# ---- test_packet ----
$(BUILDDIR)/test_packet: test_packet.c $(TCPOBJ) $(CJSONOBJ) $(LOGOBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lpthread -lm

# ---- test_math ----
$(BUILDDIR)/test_math: test_math.c $(NNSD_DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lpthread -lm

# ---- test_network ----
$(BUILDDIR)/test_network: test_network.c $(NNSD_DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lpthread -lm

# ---- test_status ----
$(BUILDDIR)/test_status: test_status.c $(NNSD_DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lpthread -lm

# ---- test_sysinfo ----
$(BUILDDIR)/test_sysinfo: test_sysinfo.c $(NNSD_DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ -lpthread -lm

ALL_TESTS = \
	$(BUILDDIR)/test_config \
	$(BUILDDIR)/test_log \
	$(BUILDDIR)/test_packet \
	$(BUILDDIR)/test_math \
	$(BUILDDIR)/test_network \
	$(BUILDDIR)/test_status \
	$(BUILDDIR)/test_sysinfo

all: $(ALL_TESTS)

run: all
	@echo "" && echo "========================================" && \
	echo "Running unit tests" && \
	echo "========================================" && echo "" && \
	failed=0; \
	for t in $(ALL_TESTS); do \
		echo "--- $$t ---"; \
		$$t; \
		if [ $$? -ne 0 ]; then failed=$$((failed+1)); fi; \
	done; \
	echo ""; \
	if [ $$failed -eq 0 ]; then \
		echo "ALL TEST SUITES PASSED"; \
	else \
		echo "$$failed TEST SUITE(S) FAILED"; \
		exit 1; \
	fi

clean:
	rm -f $(ALL_TESTS) $(NNSDOBJ)

.PHONY: all run clean
