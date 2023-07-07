ADD_EXECUTABLE(
    OGbuild
    OGbuild.c
)

TARGET_LINK_LIBRARIES(
    OGbuild
    ${ZLIB_LIBRARIES}
    marvl-db
    marvl-lib
    marvl-align
)

ADD_EXECUTABLE(
    OGlayout
    OGlayout.c
)

TARGET_LINK_LIBRARIES(
    OGlayout
    m
)

INSTALL(TARGETS OGbuild OGlayout
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

INSTALL(PROGRAMS OGtour.py tour2fasta.py colormap.py constants.py DESTINATION scripts)

