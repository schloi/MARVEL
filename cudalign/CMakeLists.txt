FIND_PACKAGE(Threads)

INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/src)

ADD_EXECUTABLE(
        cudaligner
        cuda-aligner.cpp
        debug.h debug.cpp
        cuda-match-filter.cpp cuda-match-filter.h
        ../cuda/resource-manager.cu ../cuda/resource-manager.h
        filter.c filter.h
        align.c align.h
        radix.c radix.h
        ovlbuffer.c ovlbuffer.h
        ../lib/tracks.c ../lib/tracks.h
        ../lib/compression.h ../lib/compression.c
        ../lib/read_loader.h ../lib/read_loader.c

)

SET_TARGET_PROPERTIES(
        cudaligner
        PROPERTIES CUDA_SEPARABLE_COMPILATION
        OFF
)

TARGET_INCLUDE_DIRECTORIES(
        cudaligner
        PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

TARGET_LINK_LIBRARIES(
        cudaligner
        ${ZLIB_LIBRARIES}
        marvl-db
        cuda-tracepoints
        m
        nvToolsExt
        ${CMAKE_THREAD_LIBS_INIT}

)

install(TARGETS cudaligner
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

