ADD_EXECUTABLE(
        LAcorrect
        LAcorrect.c
        consensus.h consensus.c
)

TARGET_LINK_LIBRARIES(
        LAcorrect
        ${ZLIB_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        marvl-db
        marvl-lib
        marvl-align

)


install(TARGETS LAcorrect
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

