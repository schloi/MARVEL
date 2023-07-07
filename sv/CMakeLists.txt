
ADD_EXECUTABLE(
	LAhaploscan
	LAhaploscan.c
)

TARGET_LINK_LIBRARIES(
	LAhaploscan
	${ZLIB_LIBRARIES}
	marvl-align
	marvl-db
	marvl-lib
)

ADD_EXECUTABLE(
        LAhaploval
        LAhaploval.c
)

TARGET_LINK_LIBRARIES(
        LAhaploval
        ${ZLIB_LIBRARIES}
		marvl-align
        marvl-db
        marvl-lib
)

install(TARGETS LAhaploscan LAhaploval
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
