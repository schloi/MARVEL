
FIND_PACKAGE(Threads)

ADD_LIBRARY(
        marvl-align2
        align.c
        align.h
)

ADD_EXECUTABLE(
        daligner2
        daligner.c
        filter.c filter.h
        filter_reporter_cpu.h filter_reporter_cpu.c
        ovlbuffer.h ovlbuffer.c
        align.h align.c
        radix.h radix.c
	../lib/pass.h ../lib/pass.c
	../lib/tracks.h ../lib/tracks.c
	../lib/compression.h ../lib/compression.c
)

#target_compile_options(daligner2 PUBLIC -fprofile-generate=/home/schloi)
#target_link_options(daligner2 PUBLIC -fprofile-generate=/home/schloi)

#target_compile_options(daligner2 PUBLIC -fprofile-use=/home/schloi)
#target_link_options(daligner2 PUBLIC -fprofile-use=/home/schloi)

TARGET_LINK_LIBRARIES(
        daligner2
        ${ZLIB_LIBRARIES}
        marvl-db
        m
        ${CMAKE_THREAD_LIBS_INIT}
)

ADD_EXECUTABLE(
        HPCdaligner
        HPCdaligner.c
        ../lib/tracks.c ../lib/tracks.h
        ../lib/compression.h ../lib/compression.c
)
TARGET_LINK_LIBRARIES(
        HPCdaligner
        ${ZLIB_LIBRARIES}
        marvl-db
        m
)

ADD_EXECUTABLE(
        LAcat
        LAcat.c
)
TARGET_LINK_LIBRARIES(
        LAcat
        ${ZLIB_LIBRARIES}
        marvl-db
)

ADD_EXECUTABLE(
        LAsort
        LAsort.c
)
TARGET_LINK_LIBRARIES(
        LAsort
        ${ZLIB_LIBRARIES}
        marvl-db
)

ADD_EXECUTABLE(
        LAsplit
        LAsplit.c
)
TARGET_LINK_LIBRARIES(
        LAsplit
        ${ZLIB_LIBRARIES}
        marvl-db
)


install(TARGETS LAsplit LAsort LAcat daligner2 HPCdaligner
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
