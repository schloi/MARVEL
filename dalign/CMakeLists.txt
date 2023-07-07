
ADD_LIBRARY(
        marvl-align
        align.c
        align.h
)

ADD_EXECUTABLE(
        daligner
        daligner.c
        filter.c
)

TARGET_LINK_LIBRARIES(
        daligner
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align
        m
        ${CMAKE_THREAD_LIBS_INIT}
)

install(TARGETS daligner 
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
