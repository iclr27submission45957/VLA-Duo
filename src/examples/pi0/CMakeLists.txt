# Shared PI0 code (tensor types, graph builders, forward passes), compiled once for all executables.
add_library(pi0-common OBJECT pi0-common.cpp)
target_link_libraries(pi0-common PUBLIC common mtmd)
target_compile_features(pi0-common PUBLIC cxx_std_17)

# Per-phase benchmark of NPU-only and GPU-only inference. Prints a RESULT line.
set(TARGET_BENCH llama-pi0-bench)
add_executable(${TARGET_BENCH} pi0-bench.cpp)
install(TARGETS ${TARGET_BENCH} RUNTIME)
target_link_libraries(${TARGET_BENCH} PRIVATE pi0-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_BENCH} PRIVATE cxx_std_17)

# GPU/NPU periodic pipeline over a frame stream with a two-stage cut (G->N or N->G) or a k-cut plan.
# Prints the steady-state period and latency in a RESULT line.
set(TARGET_PIPELINE llama-pi0-pipeline)
add_executable(${TARGET_PIPELINE} pi0-pipeline.cpp)
install(TARGETS ${TARGET_PIPELINE} RUNTIME)
target_link_libraries(${TARGET_PIPELINE} PRIVATE pi0-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_PIPELINE} PRIVATE cxx_std_17)

# Operator-level GPU/NPU co-execution matrix (g_ij / h_ij) used as the cut model's profile.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pi0-opmat-bench.cpp)
    set(TARGET_OPMAT_BENCH llama-pi0-opmat-bench)
    add_executable(${TARGET_OPMAT_BENCH} pi0-opmat-bench.cpp)
    install(TARGETS ${TARGET_OPMAT_BENCH} RUNTIME)
    target_link_libraries(${TARGET_OPMAT_BENCH} PRIVATE pi0-common ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET_OPMAT_BENCH} PRIVATE cxx_std_17)
endif()
