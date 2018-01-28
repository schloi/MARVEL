# Process this file with autoreconf to produce a configure script.

AC_INIT([MARVEL], [1.0], [s.schloissnig@gmail.com])
AC_PREREQ([2.69])

# AC_CONFIG_MACRO_DIR([m4])

m4_include([m4/ax_lib_hdf5.m4])
m4_include([m4/utils.m4])

AC_CONFIG_SRCDIR([corrector/consensus.c])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CC
AC_PROG_INSTALL

# check python installation

AC_ARG_WITH([python3],
    [  --with-python3=PATH     Specify the location of the python3 binary ],
    [
        PYTHON3=$withval

        if [[ ! -f $PYTHON3 ]]; then
            AC_MSG_ERROR([Invalid python3 binary specified.])
        fi
    ],
    [
        AC_PATH_PROGS([PYTHON3], [python3.6 python3 python3.5], [no])
        [if test "$PYTHON3" == "no"]
        [then]
            AM_PATH_PYTHON([3.5])
        [fi]
    ])

AC_ARG_ENABLE(versioned-dirs, [  --enable-versioned-dirs         enable versioned directories
[[default=no]]],[
  if test "$enableval" = no; then
    ac_enable_versioned=no
  else
    ac_enable_versioned=yes
  fi
  ],[ ac_enable_versioned=no])


AC_PYTHON_MODULE([networkx])

# Checks for libraries.
AX_LIB_HDF5()

AC_CHECK_LIB([m], [sin])
AC_CHECK_LIB([z], [compress])
AC_CHECK_LIB([pthread], [pthread_create])
AC_CHECK_LIB([bsd], [fgetln])

# Check for pkg-config program, used for configuring some libraries.
m4_define_default([PKG_PROG_PKG_CONFIG],
    [AC_MSG_CHECKING([pkg-config])
    AC_MSG_RESULT([no])])

PKG_PROG_PKG_CONFIG

# If the pkg-config autoconf support isn't installed, define its
# autoconf macro to disable any packages depending on it.
m4_define_default([PKG_CHECK_MODULES],
    [AC_MSG_CHECKING([$1])
    AC_MSG_RESULT([no])
    $4])

PKG_CHECK_MODULES([GTK], [gtk+-3.0], [have_gtk=yes], [have_gtk=no])

# Checks for header files.
AC_HEADER_RESOLV
AC_CHECK_HEADERS([arpa/inet.h fcntl.h limits.h locale.h netdb.h netinet/in.h stddef.h stdint.h stdlib.h string.h strings.h sys/file.h sys/param.h sys/socket.h unistd.h])

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

# generate files from templates
AC_CONFIG_FILES([Makefile
                 Makefile.settings
                 corrector/Makefile
                 dalign/Makefile
                 db/Makefile
                 lib.python/Makefile
                 lib.python/marvel/config.py
                 scripts/Makefile
                 scripts/DBprepare.py
                 scripts/paths2rids.py
                 scripts/rid2bid.py
                 scrub/Makefile
                 touring/Makefile
                 touring/OGtour.py
                 touring/tour2fasta.py
                 utils/Makefile])

AM_CONDITIONAL([HAS_MOD_EXPERIMENTAL], [test -x experimental])
AM_COND_IF([HAS_MOD_EXPERIMENTAL], [AC_CONFIG_FILES([experimental/Makefile])])

AM_CONDITIONAL([HAS_MOD_MSA], [test -x msa])
AM_COND_IF([HAS_MOD_MSA], [AC_CONFIG_FILES([msa/Makefile])])

AM_CONDITIONAL([HAS_MOD_PATCHER], [test -x patcher])
AM_COND_IF([HAS_MOD_PATCHER], [AC_CONFIG_FILES([patcher/Makefile])])

# make variables available
AC_SUBST([PYTHON3], ${PYTHON3})
AC_SUBST([have_gtk], ${have_gtk})
AC_SUBST([ac_enable_versioned], ${ac_enable_versioned})


AC_OUTPUT

# complain
if test "x${have_gtk}" = xno; then
    echo "--------------------------------"
    echo " gtk not found"
    echo " skipping apps depending on it"
    echo "--------------------------------"
fi

if test "x${with_hdf5}" = xno; then
    echo "--------------------------------"
    echo " hdf5 not found"
    echo " skipping apps depending on it"
    echo "--------------------------------"
fi

