cmake_minimum_required(VERSION 3.20)
project(nnsd C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 4060 = Ada Lovelace = sm_89
set(CMAKE_CUDA_ARCHITECTURES 89)

# Optimized build: full optimization + native SIMD (AVX2 on this machine), pthreads-safe
add_compile_options(-O3 -march=native -D_REENTRANT)

# ── cJSON ──────────────────────────────────────────────────────────────────────
add_library(cjson STATIC
    src/lib/cJSON/cJSON.c
)
target_include_directories(cjson PUBLIC src/lib/cJSON)

# ── libcsv ────────────────────────────────────────────────────────────────────
add_library(libcsv STATIC
    src/lib/libcsv/libcsv.c
)
target_include_directories(libcsv PUBLIC src/lib/libcsv)

# ── utils ─────────────────────────────────────────────────────────────────────
add_library(utils STATIC
    src/utils/tcp_utils.c
    src/utils/rest_server.c
    src/utils/log_utils.c
    src/utils/daemon.c
    src/utils/config.c
)
target_include_directories(utils PUBLIC
    src/utils
    src/h
    src/lib/cJSON
)

# ── NeuralNetInC daemon ───────────────────────────────────────────────────────────────
add_executable(nnsd
    src/nnsd/nnsd.c
    src/nnsd/nn_sysinfo.c
    src/nnsd/nn_network.c
    src/nnsd/nn_fileio.c
    src/nnsd/nn_math.c
    src/nnsd/nn_training.c
    src/nnsd/nn_packet.c
    src/nnsd/nn_cuda.cu
)
target_include_directories(nnsd PRIVATE
    src/h
    src/lib/cJSON
    src/lib/libcsv
)
target_compile_definitions(nnsd PRIVATE USE_CUDA)
target_link_directories(nnsd PRIVATE /usr/lib/x86_64-linux-gnu/stubs)
target_link_libraries(nnsd PRIVATE
    utils
    cjson
    libcsv
    pthread
    m
    cuda
    nvidia-ml
)
