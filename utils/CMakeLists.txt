ADD_EXECUTABLE(
	FAsort
	../lib/utils.h ../lib/utils.c
	FAsort.c
)

TARGET_LINK_LIBRARIES(
	FAsort
)

ADD_EXECUTABLE(
	LAcartoons
	LAcartoons.c
)

TARGET_LINK_LIBRARIES(
	LAcartoons
	m
	${ZLIB_LIBRARIES}
	marvl-align
	marvl-lib
	marvl-db
)

ADD_EXECUTABLE(
        LAshow
        ../dalign/align.c ../dalign/align.h
        LAshow.c
)

TARGET_LINK_LIBRARIES(
        LAshow
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
)

ADD_EXECUTABLE(
        LAcheck
        ../dalign/align.c
        ../dalign/align.h LAcheck.c
)

TARGET_LINK_LIBRARIES(
        LAcheck
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
)


ADD_EXECUTABLE(
        LAmerge
        LAmerge.c
        ../dalign/align.c ../dalign/align.h
        LAmergeUtils.h LAmergeUtils.c
)

TARGET_LINK_LIBRARIES(
        LAmerge
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        m
)

ADD_EXECUTABLE(
	LAcount
	LAcount.c
)

TARGET_LINK_LIBRARIES(
	LAcount
        ${ZLIB_LIBRARIES}
	marvl-align
        marvl-db
        marvl-lib
        m
)

ADD_EXECUTABLE(
	LAcoverage
	LAcoverage.c
)

TARGET_LINK_LIBRARIES(
	LAcoverage
	${ZLIB_LIBRARIES}
        marvl-align
        marvl-db
        marvl-lib
        m
)

ADD_EXECUTABLE(
	LAindex
	LAindex.c
	../lib/lasidx.h ../lib/lasidx.c
)

TARGET_LINK_LIBRARIES(
	LAindex
	marvl-align
	marvl-db
	marvl-lib
        ${ZLIB_LIBRARIES}
)

ADD_EXECUTABLE(
	LAstats
	LAstats.c
)

TARGET_LINK_LIBRARIES(
	LAstats
	marvl-align
	marvl-db
	marvl-lib
        ${ZLIB_LIBRARIES}
)

ADD_EXECUTABLE(
	TKadjust
	TKadjust.c
)

TARGET_LINK_LIBRARIES(
	TKadjust
	marvl-align
	marvl-db
	marvl-lib
        ${ZLIB_LIBRARIES}
)

ADD_EXECUTABLE(
	TKcombine
	TKcombine.c
)

TARGET_LINK_LIBRARIES(
	TKcombine
	marvl-align
	marvl-db
	marvl-lib
        ${ZLIB_LIBRARIES}
)

ADD_EXECUTABLE(
	TKmerge
	TKmerge.c
)

TARGET_LINK_LIBRARIES(
	TKmerge
	marvl-align
	marvl-db
	marvl-lib
        ${ZLIB_LIBRARIES}
)

ADD_EXECUTABLE(
	TKshow
	TKshow.c
)

TARGET_LINK_LIBRARIES(
	TKshow
	marvl-align
	marvl-db
	marvl-lib
        ${ZLIB_LIBRARIES}
)

ADD_EXECUTABLE(
	txt2track
	txt2track.c
)

TARGET_LINK_LIBRARIES(
	txt2track
	marvl-align
	marvl-db
	marvl-lib
        ${ZLIB_LIBRARIES}
)

IF (GTK_FOUND)
	ADD_EXECUTABLE(
		LAexplorer
		LAexplorer.c
			../lib/lasidx.h ../lib/lasidx.c
	)

	target_compile_options(LAexplorer PRIVATE ${GTK_CFLAGS_OTHER})
	target_include_directories(LAexplorer PRIVATE ${GTK_INCLUDE_DIRS})
	target_link_directories(LAexplorer PRIVATE ${GTK_LIBRARY_DIRS})

	TARGET_LINK_LIBRARIES(
		LAexplorer
		marvl-align
		marvl-db
		marvl-lib
			${ZLIB_LIBRARIES}
			${GTK_LIBRARIES}
	)

	install( TARGETS LAexplorer RUNTIME DESTINATION bin )
ENDIF ()

install(TARGETS txt2track TKshow TKmerge TKcombine TKadjust LAstats FAsort LAindex LAcount LAcoverage LAmerge LAcheck LAshow LAcartoons
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )
