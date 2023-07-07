ADD_LIBRARY(
        marvl-db
        OBJECT
        DB.c
        QV.c
        FA2x.h FA2x.c
        fileUtils.h fileUtils.c
)

ADD_EXECUTABLE(
        FA2db
        FA2db.c
)

TARGET_LINK_LIBRARIES(
        FA2db
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align
)

ADD_EXECUTABLE(
        DB2fa
        DB2fa.c
)

TARGET_LINK_LIBRARIES(
        DB2fa
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align
)

ADD_EXECUTABLE(
        DB2qv
        DB2qv.c
)

TARGET_LINK_LIBRARIES(
        DB2qv
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align
)

ADD_EXECUTABLE(
        DBrm
        DBrm.c
)

TARGET_LINK_LIBRARIES(
        DBrm
        ${ZLIB_LIBRARIES}
        marvl-db
)

ADD_EXECUTABLE(
        DBsplit
        DBsplit.c
)

TARGET_LINK_LIBRARIES(
        DBsplit
        ${ZLIB_LIBRARIES}
        marvl-db
)

ADD_EXECUTABLE(
        DBstats
        DBstats.c
        ../lib/utils.c
)

TARGET_LINK_LIBRARIES(
        DBstats
        m
        ${ZLIB_LIBRARIES}
        marvl-db
)

ADD_EXECUTABLE(
        DBshow
        DBshow.c
)

TARGET_LINK_LIBRARIES(
        DBshow
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align
)

ADD_EXECUTABLE(
        DBdust
        DBdust.c
)

TARGET_LINK_LIBRARIES(
        DBdust
        ${ZLIB_LIBRARIES}
        marvl-db
        m
)


install(TARGETS DBdust DBsplit DBrm DB2qv DB2fa FA2db DBstats DBshow
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

