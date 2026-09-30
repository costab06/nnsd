###### macro defines

CC        = gcc
CFLAGS    = -g -O3 -mavx2 -mfma -funroll-loops -D_REENTRANT \
            -I. -I../h -I../lib/cJSON -I../lib/libcsv
LIBS      = -lpthread -lm

# ── Optional CUDA build ───────────────────────────────────────────────────
# Build with GPU support:  make cuda   or   make USE_CUDA=1
CUDA_HOME ?= /usr/local/cuda
NVCC       = $(CUDA_HOME)/bin/nvcc
CUDA_ARCH  = -gencode arch=compute_89,code=sm_89
NVML_INC   = /usr/include
NVML_LIB   = /usr/lib/x86_64-linux-gnu

ifdef USE_CUDA
CFLAGS    += -DUSE_CUDA -I$(CUDA_HOME)/include -I$(NVML_INC)
LIBS      += -L$(NVML_LIB) -L$(NVML_LIB)/stubs -lnvidia-ml
OBJS_EXTRA = nn_cuda.o
LINK       = $(NVCC)
else
OBJS_EXTRA =
LINK       = $(CC)
endif
# ─────────────────────────────────────────────────────────────────────────

SRCS = nnsd.c nn_network.c nn_math.c nn_packet.c \
       nn_fileio.c nn_training.c nn_sysinfo.c
OBJS = $(SRCS:.c=.o)

LIB_OBJS = ../lib/cJSON/cJSON.o \
           ../lib/libcsv/libcsv.o \
           ../utils/tcp_utils.o \
           ../utils/log_utils.o \
           ../utils/daemon.o \
           ../utils/config.o \
           ../utils/rest_server.o

###### targets

default: nnsd

nnsd: $(OBJS) $(OBJS_EXTRA) $(LIB_OBJS)
	$(LINK) -o $@ $(OBJS) $(OBJS_EXTRA) $(LIB_OBJS) $(LIBS)

cuda:
	$(MAKE) USE_CUDA=1

nn_cuda.o: nn_cuda.cu ../h/nnsd.h ../h/common.h ../h/nn_cuda.h
	$(NVCC) $(CUDA_ARCH) -O2 -I. -I../h \
	        --compiler-options "-D_REENTRANT" -c $<

clean:
	rm -f nnsd $(OBJS) nn_cuda.o $(LIB_OBJS)

###### dependencies

HDRS = ../h/common.h ../h/nnsd.h ../h/packet.h ../h/nn_globals.h \
       ../h/log_utils.h ../h/nn_math.h ../h/nn_network.h \
       ../h/nn_fileio.h ../h/nn_training.h ../h/nn_packet.h \
       ../h/nn_sysinfo.h ../h/rest_server.h ../h/config.h

$(OBJS): $(HDRS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@






