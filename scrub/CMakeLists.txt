ADD_EXECUTABLE(
        LAstitch
        LAstitch.c
)

TARGET_LINK_LIBRARIES(
        LAstitch
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align
)

ADD_EXECUTABLE(
        LArepeat
        LArepeat.c
)
TARGET_LINK_LIBRARIES(
        LArepeat
        ${ZLIB_LIBRARIES}
        m
        marvl-db
        marvl-lib
        marvl-align

)

ADD_EXECUTABLE(
        LAfilter
        LAfilter.c
)
TARGET_LINK_LIBRARIES(
        LAfilter
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align

)

ADD_EXECUTABLE(
        TKhomogenize
        TKhomogenize.c
)
TARGET_LINK_LIBRARIES(
        TKhomogenize
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-lib-ext
        marvl-align

)

ADD_EXECUTABLE(
        LAq
        LAq.c
)

TARGET_LINK_LIBRARIES(
        LAq
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align

)

ADD_EXECUTABLE(
        LAgap
        LAgap.c
)

TARGET_LINK_LIBRARIES(
        LAgap
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align

)

TARGET_LINK_LIBRARIES(
        LAstitch
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align
)

ADD_EXECUTABLE(
        LAfix
        LAfix.c
)

TARGET_LINK_LIBRARIES(
        LAfix
        ${ZLIB_LIBRARIES}
        marvl-db
        marvl-lib
        marvl-align
)

install(TARGETS LAfix LAfilter LAgap LAq LArepeat LAstitch TKhomogenize
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

