CMAKE_MINIMUM_REQUIRED(VERSION 3.18 FATAL_ERROR)

SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

IF (DEFINED CMAKE_FORCE_CUDA_RELEASE)
    MESSAGE("-- Forcing CUDA compilation to Release Mode ${CMAKE_FORCE_CUDA_RELEASE}")
ENDIF ()

IF (DEFINED CMAKE_FORCE_CUDA_RELEASE OR CMAKE_BUILD_TYPE STREQUAL "Release")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --default-stream per-thread -Xptxas=-v")
ELSEIF (CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
ENDIF ()

ADD_LIBRARY(
        cuda-tracepoints
        tracepoints/definitions.h
        tracepoints/forward-wave.cuh
        tracepoints/matrix.cuh
        tracepoints/reverse-wave.cuh
        tracepoints/stream-interface.cu
        tracepoints/stream-interface.hpp
        tracepoints/tracepoints.cu
        tracepoints/tracepoints.cuh
)

TARGET_INCLUDE_DIRECTORIES(cuda-tracepoints PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

SET_TARGET_PROPERTIES(cuda-tracepoints PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

IF (APPLE)
    # We need to add the default path to the driver (libcuda.dylib) as an rpath, so that the static cuda runtime can find it at runtime.
    SET_PROPERTY(TARGET cuda-tracepoints PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
ENDIF ()
