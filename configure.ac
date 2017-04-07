#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

AC_INIT([MARVEL], [1.0], [s.schloissnig@gmail.com])

AC_CONFIG_SRCDIR([corrector/consensus.c])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_INSTALL

# Checks for libraries.
AC_CHECK_LIB([hdf5], [H5Dopen2], [have_hdf5=yes], [have_hdf5=no])
AC_CHECK_LIB([m], [sin])
AC_CHECK_LIB([z], [compress])
AC_CHECK_LIB([pthread], [pthread_create])

PKG_CHECK_MODULES([GTK], [gtk+-3.0], [have_gtk=yes], [have_gtk=no])

# Checks for header files.
AC_HEADER_RESOLV
AC_CHECK_HEADERS([arpa/inet.h fcntl.h limits.h locale.h netdb.h netinet/in.h stddef.h stdint.h stdlib.h string.h strings.h sys/file.h sys/param.h sys/socket.h unistd.h])

AC_ARG_WITH([hdf5-lib],
    [  --with-hdf5-lib=DIR     Specify the location of the hdf5 libraries ],
    [have_hdf5=yes
    HDF5_LIB=$withval])

AC_ARG_WITH(hdf5-include,
    [  --with-hdf5-include=DIR Specify the location of the hdf5 include files ],
    [have_hdf5=yes
    HDF5_INC=$withval])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_OFF_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_FSEEKO
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([bzero floor ftruncate gethostbyname inet_ntoa memmove memset mkdir pow realpath select setlocale socket sqrt strcasecmp strchr strdup strerror strncasecmp strrchr strtol])

AC_CONFIG_FILES([Makefile
                 Makefile.settings
                 corrector/Makefile
                 dalign/Makefile
                 db/Makefile
                 lib.python/Makefile
                 lib.python/marvel/config.py
                 scripts/Makefile
                 scrub/Makefile
                 touring/Makefile
                 utils/Makefile])

AC_SUBST([have_gtk], ${have_gtk})
AC_SUBST([have_hdf5], ${have_hdf5})
AC_SUBST(HDF5_INC)
AC_SUBST(HDF5_LIB)

AC_OUTPUT

if test "x${have_gtk}" = xno; then
    echo "--------------------------------"
    echo " gtk not found"
    echo " skipping apps depending on it"
    echo "--------------------------------"
fi

if test "x${have_hdf5}" = xno; then
    echo "--------------------------------"
    echo " hdf5 not found"
    echo " skipping apps depending on it"
    echo "--------------------------------"
fi


